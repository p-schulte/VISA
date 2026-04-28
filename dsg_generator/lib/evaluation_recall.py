import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from lib.ults.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps

class BasicSceneGraphEvaluator:
    def __init__(self, mode, BW_object_classes, BW_all_predicates, BW_spatial_predicates,
                 iou_threshold=0.5, save_file="tmp",constraint=False, semithreshold=None):
        self.result_dict = {}
        self.mode = mode
        self.num_rel = len(BW_all_predicates)
        self.result_dict[self.mode + '_accuracy'] = {"tp": 0, "fp": 0, "fn": 0}
        self.result_dict[self.mode + '_per_rel_accuracy'] = {
            "tp": np.zeros(self.num_rel, dtype=int),
            "fp": np.zeros(self.num_rel, dtype=int),
            "fn": np.zeros(self.num_rel, dtype=int),
        }

        self.result_dict[self.mode + '_recall'] = {10: [], 15:[], 20: [], 50: [], 100: []}  
        self.result_dict[self.mode + '_mean_recall_collect'] = {10: [[] for i in range(self.num_rel)], 15: [[] for i in range(self.num_rel)], 20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.constraint = constraint # semi constraint if True
        self.iou_threshold = iou_threshold
        self.BW_object_classes = BW_object_classes
        self.BW_all_predicates = BW_all_predicates
        self.BW_spatial_predicates = BW_spatial_predicates
        self.semithreshold = semithreshold
        self.save_file = save_file
        with open(self.save_file, "w") as f:
            f.write("Begin training\n")

    def reset_result(self):
        self.result_dict[self.mode + '_accuracy'] = {"tp": 0, "fp": 0, "fn": 0}
        self.result_dict[self.mode + '_per_rel_accuracy'] = {
            "tp": np.zeros(self.num_rel, dtype=int),
            "fp": np.zeros(self.num_rel, dtype=int),
            "fn": np.zeros(self.num_rel, dtype=int),
        }

        self.result_dict[self.mode + '_recall'] = {10: [], 15:[], 20: [], 50: [], 100: []}  
        self.result_dict[self.mode + '_mean_recall_collect'] = {10: [[] for i in range(self.num_rel)], 15: [[] for i in range(self.num_rel)], 20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}

    def print_stats(self):
        with open(self.save_file, "a") as f:
            f.write('======================' + self.mode + '============================\n')
            print('======================' + self.mode + '============================')
            print("Recall@K:")
            for k, v in self.result_dict[self.mode + '_recall'].items():
                print('R@%i: %f' % (k, np.mean(v)))
                f.write('R@%i: %f\n' % (k, np.mean(v)))
                        
            print()
            print("MeanRecall@K:")
            for k, v in self.result_dict[self.mode + '_mean_recall_collect'].items():
                sum_recall = 0
                for idx in range(self.num_rel):
                    if len(v[idx]) == 0:
                        tmp_recall = 0.0
                    else:
                        tmp_recall = np.mean(v[idx])
                    sum_recall += tmp_recall

                print('R@%i: %f' % (k, sum_recall / float(self.num_rel)))
                f.write('R@%i: %f\n' % (k, sum_recall / float(self.num_rel)))

            # own metrics
            print()
            print("Own metrics:")
            tp = self.result_dict[self.mode + '_accuracy']["tp"]
            fp = self.result_dict[self.mode + '_accuracy']["fp"]
            fn = self.result_dict[self.mode + '_accuracy']["fn"]
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            try:
                f1 = 2 * (precision * recall) / (precision + recall)
            except:
                f1 = 0
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1: {f1}')
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')
            f.write(f'F1: {f1}\n')


            print()
            print("Per-relation metrics:")
            per_rel = self.result_dict[self.mode + '_per_rel_accuracy']
            tp_rel = per_rel["tp"]
            fp_rel = per_rel["fp"]
            fn_rel = per_rel["fn"]

            f1_list = []   # <--- collect per-relation F1 here

            for rel_id in range(self.num_rel):
                tp = tp_rel[rel_id]
                fp = fp_rel[rel_id]
                fn = fn_rel[rel_id]

                if tp + fp == 0 and tp + fn == 0:
                    precision = recall = f1 = 0.0
                else:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0

                # only count relations that actually appear, if you want
                if tp + fp + fn > 0:
                    f1_list.append(f1)

                rel_name = (
                    self.BW_all_predicates[rel_id]
                    if hasattr(self, "BW_all_predicates") and rel_id < len(self.BW_all_predicates)
                    else str(rel_id)
                )

                print(f"Rel {rel_id} ({rel_name}): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
                f.write(f"Rel {rel_id} ({rel_name}): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}\n")

            # ---- relation-averaged F1 (macro over relations) ----
            if len(f1_list) > 0:
                macro_f1 = float(np.mean(f1_list))
            else:
                macro_f1 = 0.0

            print(f"\nRelation-averaged F1 (macro over predicates): {macro_f1:.4f}")
            f.write(f"\nRelation-averaged F1 (macro over predicates): {macro_f1:.4f}\n")


                

    def evaluate_scene_graph(self, gt, pred):
        '''collect the groundtruth and prediction'''
        pred['spatial_distribution'] = torch.sigmoid(pred['spatial_distribution'])
        detection_counter= 0
        for idx, frame_gt in enumerate(gt):
            frame_gt = frame_gt[1:] # remove the first element which is the frame name

            # 1. Create Ground Truth
            gt_boxes = np.zeros([len(frame_gt), 4])  # No human, just objects
            gt_classes = np.zeros(len(frame_gt))
            gt_relations = []
            
            for m, n in enumerate(frame_gt):
                # each pair
                gt_boxes[m,:] = n['bbox']
                gt_classes[m] = n['class']
            

            # Create all pairwise relationships
            for i1, obj1 in enumerate(frame_gt):
                for i2, obj2 in enumerate(frame_gt):
                    obj1 = frame_gt[i1]
                    obj2 = frame_gt[i2]

                    obj2_id = obj2['identifier']  # Identifier for obj2

                    # --- Check Spatial Relationships (Object-Object) ---
                    spatial_rel_tensors, spatial_rel_ids = obj1['spatial_relationship']
                    for rel_idx, related_id in zip(spatial_rel_tensors.tolist(), spatial_rel_ids):
                        if related_id == obj2_id:  # Ensure correct identifier
                            gt_relations.append([i1, i2, rel_idx])
                    

            gt_entry = {
                'gt_classes': gt_classes,
                'gt_relations': np.array(gt_relations),
                'gt_boxes': gt_boxes,
            }
            
            rels_i = pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()
            rels_i -= detection_counter
            import math
            num_predictions = int(math.sqrt((pred['im_idx'] == idx).sum().item()))
            detection_counter += num_predictions # if the number of detections is different from the number of ground truth elements

            frame_box_mask = pred['boxes'][:, 0] == idx
            pred_entry = {
                'pred_boxes': pred['boxes'][frame_box_mask, 1:].cpu().clone().numpy(),
                'pred_classes': pred['pred_labels'][frame_box_mask].cpu().clone().numpy(),
                'pred_rel_inds': rels_i,
                'obj_scores': pred['pred_scores'][frame_box_mask].cpu().clone().numpy(),
                'rel_scores': pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy()
            }

            evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict,
                               iou_thresh=self.iou_threshold, method=self.constraint, threshold=self.semithreshold, num_rel= self.num_rel)

def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, method=None, threshold = 0.9, num_rel=26, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param result_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']


    pred_boxes = pred_entry['pred_boxes'].astype(float)
    pred_classes = pred_entry['pred_classes']
    obj_scores = pred_entry['obj_scores']
   
    if method == 'semi':
        pred_rels = []
        predicate_scores = []
        for i, j in enumerate(pred_rel_inds):
            # this is the spatial distribution
            for k in np.where(rel_scores[i] > threshold)[0]:
                pred_rels.append(np.append(j, k))
                predicate_scores.append(rel_scores[i, k])
        pred_rels = np.array(pred_rels)
        predicate_scores = np.array(predicate_scores)
    elif method == 'no':
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]
    else:
        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1)))
        predicate_scores = rel_scores.max(1)

    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
                gt_rels, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes,
                predicate_scores, obj_scores, phrdet= mode=='phrdet',
                **kwargs)
    
        # --- existing global metrics ---
    match = reduce(np.union1d, pred_to_gt)  # get all matched predictions
    tp = len(match)
    fp = len(pred_to_gt) - len(match)
    fn = gt_rels.shape[0] - len(match)
    result_dict[mode + '_accuracy']["tp"] += tp
    result_dict[mode + '_accuracy']["fp"] += fp
    result_dict[mode + '_accuracy']["fn"] += fn

    # --- NEW: per-relation metrics ---
    per_rel_tp = np.zeros(num_rel, dtype=int)
    per_rel_fp = np.zeros(num_rel, dtype=int)
    per_rel_fn = np.zeros(num_rel, dtype=int)

    # 1) TPs and FPs per relation (iterate over predictions)
    num_pred = len(pred_to_gt)
    for pred_idx in range(num_pred):
        rel_label = int(pred_rels[pred_idx, 2])  # predicate class id
        if len(pred_to_gt[pred_idx]) > 0:
            per_rel_tp[rel_label] += 1
        else:
            per_rel_fp[rel_label] += 1

    # 2) FNs per relation (GT relations not matched by any prediction)
    gt_matched = np.zeros(gt_rels.shape[0], dtype=bool)
    for pred_idx in range(num_pred):
        for gt_ind in pred_to_gt[pred_idx]:
            gt_matched[gt_ind] = True

    for gt_ind, matched in enumerate(gt_matched):
        if not matched:
            rel_label = int(gt_rels[gt_ind, 2])
            per_rel_fn[rel_label] += 1

    # 3) Accumulate into result_dict
    result_dict[mode + '_per_rel_accuracy']["tp"] += per_rel_tp
    result_dict[mode + '_per_rel_accuracy']["fp"] += per_rel_fp
    result_dict[mode + '_per_rel_accuracy']["fn"] += per_rel_fn




    for k in result_dict[mode + '_recall']:
        match = reduce(np.union1d, pred_to_gt[:k])
        recall_hit = [0] * num_rel
        recall_count = [0] * num_rel
        
        for idx in range(gt_rels.shape[0]):
            local_label = gt_rels[idx,2]
            recall_count[int(local_label)] += 1

        for idx in range(len(match)): 
            local_label = gt_rels[int(match[idx]),2]
            recall_hit[int(local_label)] += 1
        
        for n in range(num_rel):
            if recall_count[n] > 0:
                result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
                    
        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)

    # Calculate own eval metrics
    match = reduce(np.union1d, pred_to_gt) # get all matched predictions
    tp = len(match)
    fp = len(pred_to_gt) - len(match)
    fn = gt_rels.shape[0] - len(match)
    result_dict[mode + '_accuracy']["tp"]+= tp
    result_dict[mode + '_accuracy']["fp"]+= fp
    result_dict[mode + '_accuracy']["fn"]+= fn

    COMPUTE_METS = False
    if COMPUTE_METS:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
        import pdb;pdb.set_trace()
    
    return pred_to_gt, pred_5ples, rel_scores

###########################
def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    sorted_scores = relation_scores.prod(1)
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
    scores_overall = relation_scores.prod(1)

    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
