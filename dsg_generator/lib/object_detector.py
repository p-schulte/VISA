import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import cv2
import os
from fasterRCNN.lib.model.utils.config import cfg

from lib.funcs import assign_relations, assign_relations_bw
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fasterRCNN.lib.model.roi_layers import nms

class detector(nn.Module):

    '''first part: object detection (image/video)'''

    def __init__(self, train, object_classes, use_SUPPLY, model_path, mode='sgdet'):
        super(detector, self).__init__()

        # --- small-box filtering (generic) ---
        # boxes smaller than this fraction of image area are considered "tiny"
        self.min_box_area_ratio = 0.005  # 0.5% of image area – tune if needed

        # start in "debug mode": just print small boxes, don't remove them
        self.filter_small_boxes = False




        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode

        self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False, use_for_dsg=True)
        self.fasterRCNN.create_architecture()
        self.model_path = model_path
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.fasterRCNN.load_state_dict(checkpoint['model'])

        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):

        counter = 0
        counter_image = 0

        # create saved-bbox, labels, scores, features
        import torch
        FINAL_BBOXES = torch.tensor([]).to(self.device)
        FINAL_LABELS = torch.tensor([], dtype=torch.int64).to(self.device)
        FINAL_SCORES = torch.tensor([]).to(self.device)
        FINAL_FEATURES = torch.tensor([]).to(self.device)
        FINAL_BASE_FEATURES = torch.tensor([]).to(self.device)

        while counter < im_data.shape[0]:
            if counter + 10 < im_data.shape[0]:
                inputs_data = im_data[counter:counter + 10]
                inputs_info = im_info[counter:counter + 10]
                inputs_gtboxes = gt_boxes[counter:counter + 10]
                inputs_numboxes = num_boxes[counter:counter + 10]

            else:
                inputs_data = im_data[counter:]
                inputs_info = im_info[counter:]
                inputs_gtboxes = gt_boxes[counter:]
                inputs_numboxes = num_boxes[counter:]

            rois, cls_prob, bbox_pred, base_feat, roi_features = self.fasterRCNN(inputs_data, inputs_info,
                                                                                    inputs_gtboxes, inputs_numboxes)

            SCORES = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            # bbox regression (class specific)
            box_deltas = bbox_pred.data
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(self.device) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(self.device)
            box_deltas = box_deltas.view(-1, rois.shape[1], 4 * len(self.object_classes))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            PRED_BOXES = clip_boxes(pred_boxes, im_info.data, 1)

            PRED_BOXES /= inputs_info[0, 2]

            #traverse frames
            for i in range(rois.shape[0]):
                # images in the batch
                scores = SCORES[i]
                pred_boxes = PRED_BOXES[i]

                for j in range(1, len(self.object_classes)):
                    # NMS according to obj categories
                    inds = torch.nonzero(scores[:, j] > cfg.TEST.SCORE_THRESHOLD).view(-1) #0.05 is score threshold
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS) # NMS threshold
                        cls_dets = cls_dets[keep.view(-1).long()]



                        # -------------------------------
                        # generic tiny-box detection
                        # -------------------------------
                        if cls_dets.numel() > 0:
                            # image size (in original scale)
                            img_h = inputs_info[i, 0].item()
                            img_w = inputs_info[i, 1].item()
                            img_area = img_h * img_w
                            min_area = self.min_box_area_ratio * img_area

                            boxes_only = cls_dets[:, 0:4]
                            wh = boxes_only[:, 2:4] - boxes_only[:, 0:2]
                            areas = wh[:, 0] * wh[:, 1]
                            small_mask = areas < min_area  # True = tiny box

                            if small_mask.any():
                                # boxes that would be filtered
                                tiny_boxes = boxes_only[small_mask].detach().cpu().numpy()
                                tiny_scores = cls_dets[small_mask, 4].detach().cpu().numpy()
                                # print(f"[DEBUG] frame {counter_image}, class {j}: "
                                #    f"{tiny_boxes.shape[0]} tiny boxes below min_area={min_area:.1f}")
                                for b, s in zip(tiny_boxes, tiny_scores):
                                    # ensure numeric formatting works even if b is a numpy array
                                    try:
                                        area = float((b[2] - b[0]) * (b[3] - b[1]))
                                    except Exception:
                                        area = float(np.prod(np.array(b)[2:4] - np.array(b)[0:2]))
                                    b_str = np.array2string(np.array(b), precision=1)
                                    # print(f"    bbox={b_str}, score={float(s):.3f}, area={area:.1f}")

                                # when filter_small_boxes is True, actually drop them
                                if self.filter_small_boxes:
                                    keep_big = ~small_mask
                                    cls_dets = cls_dets[keep_big]
                                    # also restrict 'keep' so ROI features stay aligned
                                    keep = keep[keep_big]

                        # if everything got filtered, skip this class
                        if cls_dets.numel() == 0:
                            continue






                        
                        final_bbox = cls_dets[:, 0:4]
                        final_score = cls_dets[:, 4]
                        final_labels = torch.tensor([j]).repeat(keep.shape[0]).to(self.device)
                        final_features = roi_features[i, inds[order[keep]]]

                        final_bbox = torch.cat((torch.tensor([[counter_image]], dtype=torch.float).repeat(final_bbox.shape[0], 1).to(self.device),
                                                final_bbox), 1)
                        FINAL_BBOXES = torch.cat((FINAL_BBOXES, final_bbox), 0)
                        FINAL_LABELS = torch.cat((FINAL_LABELS, final_labels), 0)
                        FINAL_SCORES = torch.cat((FINAL_SCORES, final_score), 0)
                        FINAL_FEATURES = torch.cat((FINAL_FEATURES, final_features), 0)
                FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat[i].unsqueeze(0)), 0)

                counter_image += 1

            counter += 10
        FINAL_BBOXES = torch.clamp(FINAL_BBOXES, 0)                      
        prediction = {'FINAL_BBOXES': FINAL_BBOXES, 'FINAL_LABELS': FINAL_LABELS, 'FINAL_SCORES': FINAL_SCORES,
                        'FINAL_FEATURES': FINAL_FEATURES, 'FINAL_BASE_FEATURES': FINAL_BASE_FEATURES}
        


        # import pdb;pdb.set_trace()


        if self.is_train:
            DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations_bw(prediction, gt_annotation, assign_IOU_threshold=0.5)




            if self.use_SUPPLY: 
                FINAL_BBOXES_X = torch.tensor([]).to(self.device)
                FINAL_LABELS_X = torch.tensor([], dtype=torch.int64).to(self.device)
                FINAL_SCORES_X = torch.tensor([]).to(self.device)
                FINAL_FEATURES_X = torch.tensor([]).to(self.device)

                assigned_labels = torch.tensor(assigned_labels, dtype=torch.long).to(FINAL_BBOXES_X.device)

                for i, missing_objects in enumerate(SUPPLY_RELATIONS):

                    # clean the detection ids and bounding boxes
                    del_set_indices = []        # 1. remove all incorrect bounding boxes (+save removed indices)
                    initial_length = len(FINAL_BBOXES)
                    for z in range(np.max(DETECTOR_FOUND_IDX[i])):
                        if z not in DETECTOR_FOUND_IDX[i]:
                            del_set_indices.append(z)
                    if del_set_indices != []:
                        offset = torch.where(FINAL_BBOXES[:, 0] == i)[0][0].item()
                        del_set_proper_indices = [x + offset for x in del_set_indices]
                        del_set_proper_indices.sort(reverse=True)
                        for rem_ind in del_set_proper_indices:
                            FINAL_BBOXES = torch.cat([FINAL_BBOXES[:rem_ind], FINAL_BBOXES[rem_ind + 1:]])
                            assigned_labels = torch.cat([assigned_labels[:rem_ind], assigned_labels[rem_ind + 1:]])
                            FINAL_SCORES = torch.cat([FINAL_SCORES[:rem_ind], FINAL_SCORES[rem_ind + 1:]])
                            FINAL_LABELS = torch.cat([FINAL_LABELS[:rem_ind], FINAL_LABELS[rem_ind + 1:]])
                            FINAL_FEATURES = torch.cat([FINAL_FEATURES[:rem_ind], FINAL_FEATURES[rem_ind + 1:]])
                    assert len(FINAL_BBOXES) == initial_length - len(del_set_indices), "Length of FINAL_BBOXES is not correct after deletion"
                    
                    del_set_indices.sort(reverse=True)
                    for del_index in del_set_indices:      # 2. decrease all indices of detection ids above the removed ones
                        for z in range(len(DETECTOR_FOUND_IDX[i])):
                            if DETECTOR_FOUND_IDX[i][z] > del_index:
                                DETECTOR_FOUND_IDX[i][z] -= 1



                    if len(missing_objects) > 0:
                        # Allocate space for unfound ground-truth objects
                        unfound_gt_bboxes = torch.zeros([len(missing_objects), 5]).to(self.device)
                        unfound_gt_classes = torch.zeros([len(missing_objects)], dtype=torch.int64).to(self.device)
                        one_scores = torch.ones([len(missing_objects)], dtype=torch.float32).to(self.device)  # probability 1

                        for m, obj in enumerate(missing_objects):
                            if 'bbox' in obj.keys():
                                # Store the missing object box and scale it back
                                unfound_gt_bboxes[m, 1:] = torch.tensor(obj['bbox']).to(self.device) * im_info[i, 2]
                                unfound_gt_classes[m] = obj['class']
                            else:
                                continue 

                        # Update DETECTOR_FOUND_IDX by adding indices of supplied objects
                        start_idx = int(sum(FINAL_BBOXES[:, 0] == i))
                        end_idx = start_idx + len(SUPPLY_RELATIONS[i])
                        DETECTOR_FOUND_IDX[i] = list(np.concatenate((DETECTOR_FOUND_IDX[i], np.arange(start=start_idx, stop=end_idx)), axis=0).astype('int64'))

                        # Augment ground-truth relationships with supplied objects
                        GT_RELATIONS[i].extend(SUPPLY_RELATIONS[i])

                        # Compute features for supplied objects using RoI Align
                        pooled_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES[i].unsqueeze(0), unfound_gt_bboxes.to(self.device))
                        pooled_feat = self.fasterRCNN._head_to_tail(pooled_feat)

                        # Normalize bbox coordinates
                        unfound_gt_bboxes[:, 0] = i  # Assign frame index
                        unfound_gt_bboxes[:, 1:] = unfound_gt_bboxes[:, 1:] / im_info[i, 2]

                        # Concatenate the new objects into the final tensors
                        FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i], unfound_gt_bboxes))
                        FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i], unfound_gt_classes))
                        FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i], one_scores))
                        FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i], pooled_feat))

                    else:
                        # No missing objects, just copy existing ones
                        FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i]))
                        FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i]))
                        FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i]))
                        FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i]))


            FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES_X)[:, 1:], dim=1)
            global_idx = torch.arange(start=0, end=FINAL_BBOXES_X.shape[0])  # all bbox indices
            device = FINAL_BBOXES_X.device  # Get the device of FINAL_BBOXES_X
            global_idx = global_idx.to(device)  # Move global_idx to the same device

            im_idx = []  # which frame are the relations belong to
            pair = []
            s_rel = []
            for i, j in enumerate(DETECTOR_FOUND_IDX): # DETECTOR IS A LIST FOR EACH FRAME, AND IN EACH FRAME, THE INDEX I DESCRIBES THE INDEX OF THE DETECTIONS WHERE THIS OBJECT IS
                for obj1 in range(len(GT_RELATIONS[i])):  # Iterate over all detected objects
                    for obj2 in range(len(GT_RELATIONS[i])):  # Include each object with itself (n^2)
                        im_idx.append(i)  # Image index
                        o1 = int(global_idx[FINAL_BBOXES_X[:, 0] == i][obj1])                            
                        o2 = int(global_idx[FINAL_BBOXES_X[:, 0] == i][obj2])
                        pair.append([o1, o2])  # Object-Object pairing

                        r_index_o1 = DETECTOR_FOUND_IDX[i].index(obj1)
                        r_index_o2 = DETECTOR_FOUND_IDX[i].index(obj2)                        
                        
                        spat_rel_names, spat_rel_args = GT_RELATIONS[i][r_index_o1].get('spatial_relationship', ([], []))
                        tmp_list = []
                        for p in range(len(spat_rel_names)):
                            if GT_RELATIONS[i][r_index_o2]['identifier'] == spat_rel_args[p]:
                                tmp_list.append(spat_rel_names[p].item())
                        s_rel.append(tmp_list)


            pair = torch.tensor(pair).to(self.device)
            im_idx = torch.tensor(im_idx, dtype=torch.float).to(self.device)
            union_boxes = torch.cat((im_idx[:, None],
                                        torch.min(FINAL_BBOXES_X[:, 1:3][pair[:, 0]],
                                                FINAL_BBOXES_X[:, 1:3][pair[:, 1]]),
                                        torch.max(FINAL_BBOXES_X[:, 3:5][pair[:, 0]],
                                                FINAL_BBOXES_X[:, 3:5][pair[:, 1]])), 1)

            union_boxes[:, 1:] = union_boxes[:, 1:] * im_info[0, 2]
            union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)

            pair_rois = torch.cat((FINAL_BBOXES_X[pair[:,0],1:],FINAL_BBOXES_X[pair[:,1],1:]), 1).data.cpu().numpy()
            spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

            entry = {'boxes': FINAL_BBOXES_X,
                        'labels': FINAL_LABELS_X,
                        'scores': FINAL_SCORES_X,
                        'distribution': FINAL_DISTRIBUTIONS,
                        'im_idx': im_idx,
                        'pair_idx': pair,
                        'features': FINAL_FEATURES_X,
                        'union_feat': union_feat,
                        'spatial_masks': spatial_masks,
                        'spatial_gt': s_rel}

            return entry

        else:                
            DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations_bw(prediction, gt_annotation, assign_IOU_threshold=0.3)
            FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
            FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
            PRED_LABELS = PRED_LABELS + 1


            entry = {'boxes': FINAL_BBOXES,
                        'labels': torch.LongTensor(assigned_labels).to(self.device),
                        'scores': FINAL_SCORES,
                        'distribution': FINAL_DISTRIBUTIONS,
                        'pred_labels': PRED_LABELS,
                        'features': FINAL_FEATURES,
                        'fmaps': FINAL_BASE_FEATURES,
                        'im_info': im_info[0, 2]}

            return entry
        