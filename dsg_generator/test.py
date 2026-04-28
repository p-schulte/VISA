import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
from time import time


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_loader import CONFIG

from lib.config import Config as STTranConfig
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.sttran import STTran
from lib.track import get_sequence, get_sequence_simple
from lib.matcher import *
sttran_config = STTranConfig()
for i in sttran_config.args:
    print(i,':', sttran_config.args[i])


# load paths from config file
config = CONFIG.dsg
DATA_PATH = config['data_path']
DETECTOR_MODEL_PATH = config['detector_model_path']
DSG_MODEL_PATH = config['dsg_model_path']


# Own dataset (BlocksWorld--pddl-format):
from dataloader.blocksworld import Blocksworld as BW, cuda_collate_fn as ccf

# training set
train_BW_dataset = BW(mode="train", datasize=sttran_config.datasize, data_path=DATA_PATH)
train_dataloader = torch.utils.data.DataLoader(train_BW_dataset, shuffle=False, num_workers=4,
                                              collate_fn=ccf, pin_memory=False)

# testing set
test_BW_dataset = BW(mode="test", datasize=sttran_config.datasize, data_path=DATA_PATH)
test_dataloader = torch.utils.data.DataLoader(test_BW_dataset, shuffle=False, num_workers=4,
                                              collate_fn=ccf, pin_memory=False)

dloaders = [train_dataloader, test_dataloader]
sets = [train_BW_dataset, test_BW_dataset]
for dataloader, BW_dataset in zip(dloaders, sets):
    set_name = BW_dataset.mode
    print()
    print()
    print('#'*50)
    print(f"{set_name.upper()} SET")
    print('#'*50)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # freeze the detection backbone
    object_detector = detector(train=False, object_classes=BW_dataset.object_classes, use_SUPPLY=True, model_path=DETECTOR_MODEL_PATH, mode=sttran_config.mode).to(device=device)
    object_detector.eval()


    model = STTran(mode=sttran_config.mode,
                spatial_class_num=len(BW_dataset.spatial_relationships),
                obj_classes=BW_dataset.object_classes,
                enc_layer_num=sttran_config.enc_layer,
                dec_layer_num=sttran_config.dec_layer).to(device=device)

    model.eval()

    ckpt = torch.load(DSG_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print('*'*50)
    #print('CKPT {} is loaded'.format(sttran_config.model_path))
    #

    evaluator = BasicSceneGraphEvaluator( # using a threshold to filter out low confidence predictions
        mode=sttran_config.mode,
        BW_object_classes=BW_dataset.object_classes,
        BW_all_predicates=BW_dataset.relationship_classes,
        BW_spatial_predicates=BW_dataset.spatial_relationships,
        iou_threshold=0.5,
        constraint='semi', semithreshold=0.6)

    matcher= HungarianMatcher(0.5, 1, 1, 0.5)
    matcher.eval()
    all_time = []
    with torch.no_grad():
        for b, data in enumerate(dataloader):
            start = time()
            im_data = copy.deepcopy(data[0]).to(device)
            im_info = copy.deepcopy(data[1]).to(device)
            gt_boxes = copy.deepcopy(data[2]).to(device)
            num_boxes = copy.deepcopy(data[3]).to(device)
            gt_annotation = BW_dataset.gt_annotations[data[4]]

            with torch.no_grad():
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            get_sequence(entry, gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, sttran_config.mode, GENERAL_COARSE_TRACKING=True)
            pred = model(entry)
            all_time.append(time()-start)



            # visualize prediction
            VIZ = False
            if VIZ:
                import lib.visualize_predictions as viz_pred

                # Visualize detected objects with regard to assigned tracklet
                REFINE_IDS = True # if this is set to true, the indices of the objects in the images are not the detection-ids but the tracklet-ids
                if REFINE_IDS:
                    get_sequence(pred, gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, sttran_config.mode, GENERAL_COARSE_TRACKING=True)

                    box_to_tracklet = list()
                    indices = pred['indices']
                    for i in range(len(pred['boxes'])):
                        index = -1
                        for i_t, tracklet in enumerate(indices):
                            if i in tracklet:
                                index = i_t
                                break
                        box_to_tracklet.append(index)
                    

                for frame_idx in range(len(gt_annotation)):

                    # Only use correct part of box_to_tracklet
                    import copy
                    start = (pred['boxes'][:, 0] == frame_idx).nonzero(as_tuple=True)[0][0].item()
                    end = (pred['boxes'][:, 0] == frame_idx).nonzero(as_tuple=True)[0][-1].item()
                    btt_excerpt = copy.deepcopy(box_to_tracklet[start:end+1]) if REFINE_IDS else None
                    
                    viz_pred.visualize_object_detection(im_data, pred, BW_dataset, frame_idx=frame_idx, filename=f"viz/det_{b}_{frame_idx}.png", box_to_tracklet = btt_excerpt)
                    viz_pred.visualize_scene_graph(pred, BW_dataset, f"viz/dsg_{b}_{frame_idx}.png", frame_idx=frame_idx, thresh=0.1, box_to_tracklet = btt_excerpt)

                import pdb; pdb.set_trace()
                
            evaluator.evaluate_scene_graph(gt_annotation, dict(pred))
    print('Averge inference time', np.mean(all_time))
            
    print('-------------------------semi constraint-------------------------------')
    evaluator.print_stats()