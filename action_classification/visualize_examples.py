import torch
from torch.utils.data import DataLoader
from lib.net.stgcn import STGCN
from lib.dataloader.dataloader import BlocksworldSequences, cuda_collate_fn as ccf
from lib.tools.dsg_construction_utils import construct_dynamic_scene_graph, construct_dynamic_scene_graph_with_tracking
import torch.nn as nn
import numpy as np

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dsg_generator.lib.visualize_predictions import visualize_object_detection, visualize_predictions_console, visualize_scene_graph
from dsg_generator.lib.track import get_sequence, get_sequence_simple
from dsg_generator.lib.sttran import STTran
from dsg_generator.lib.AdamW import AdamW

from dsg_generator.lib.matcher import HungarianMatcher
from dsg_generator.lib.object_detector import detector
from dsg_generator.lib.config import Config as STTranConfig
from config.config_loader import CONFIG
from action_classification.train import run_epoch, custom_cuda_collate_fn


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    conf = CONFIG.ac
    num_epochs = conf['num_epochs']
    batch_size = conf['batch_size']
    #spatial_kernel_size = conf['spatial_kernel_size'][CONFIG.dsg['DATASET_NAME']]
    temporal_kernel_size = conf['temporal_kernel_size']
    num_nodes = conf['num_nodes'][CONFIG.dsg['DATASET_NAME']]
    num_features = conf['num_features']
    num_example_visualization_instances = CONFIG.dsg['num_example_visualization_instances']
    _arg_method = conf['stgcn_model']['argument_prediction']['method']
    optimizer = conf['stgcn_model']['argument_prediction']['parameters'][_arg_method]['optimizer']
    use_lr_scheduler = conf['use_lr_scheduler']
    use_early_stopping = conf['use_early_stopping']
    early_stopping_patience = conf['early_stopping_patience']
    early_stopping_min_delta = conf['early_stopping_min_delta']
    print(f"Optimizer: {optimizer}")
    if optimizer == 'Adam':
        adam_lr = conf['optimizer_params']['Adam']['learning_rate']
        adam_wd = conf['optimizer_params']['Adam']['weight_decay']
    elif optimizer == 'AdamW':
        adamw_lr = conf['optimizer_params']['AdamW']['learning_rate']
        adamw_wd = conf['optimizer_params']['AdamW']['weight_decay']
    elif optimizer == 'SGD':
        sgd_lr = conf['optimizer_params']['SGD']['learning_rate']
        sgd_momentum = conf['optimizer_params']['SGD']['momentum']
        sgd_wd = conf['optimizer_params']['SGD']['weight_decay']
    elif optimizer == 'RMSprop':
        rmsprop_lr = conf['optimizer_params']['RMSprop']['learning_rate']
        rmsprop_alpha = conf['optimizer_params']['RMSprop']['alpha']
        rmsprop_wd = conf['optimizer_params']['RMSprop']['weight_decay']



    
    # load paths from config file
    config = CONFIG.dsg
    DATASET_DIR = config['data_path']
    OBJ_DET_MODEL_PATH = config['detector_model_path']
    DSG_MODEL_PATH = config['dsg_model_path']
    AC_MODEL_PATH = config['ac_model_path']
    

    # Loading the datasets
    train_dataset = BlocksworldSequences( # train dataset
        mode = 'train',
        root_dir=DATASET_DIR
        )
    # train_loader = DataLoader(train_dataset, collate_fn=custom_cuda_collate_fn, batch_size=batch_size, shuffle=True)

    test_dataset = BlocksworldSequences( # test dataset
        mode = 'test',
        root_dir=DATASET_DIR
        )
    # test_loader = DataLoader(test_dataset, collate_fn=custom_cuda_collate_fn, batch_size=batch_size, shuffle=False)

    # set spatial kernel size via number of relationships:
    spatial_kernel_size = len(train_dataset.spatial_relationships)+1  # +1 for self-loop

    from torch.utils.data import Subset
    import random
    # Random subset indices
    train_indices = random.sample(range(len(train_dataset)), num_example_visualization_instances)
    test_indices = random.sample(range(len(test_dataset)), num_example_visualization_instances)


    # Wrap datasets with Subset
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    # New loaders
    train_loader = DataLoader(train_subset, collate_fn=custom_cuda_collate_fn, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, collate_fn=custom_cuda_collate_fn, batch_size=batch_size, shuffle=False)


    # Model, optimizer, loss function, and device
    model = STGCN(
        num_classes=train_dataset.num_classes,
        in_channels=num_features,
        spatial_kernel_size=spatial_kernel_size,
        temporal_kernel_size=temporal_kernel_size,
        num_nodes=num_nodes
    )
    model.eval()
    ckpt = torch.load(AC_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print()
    print('*'*50)
    print('AC checkpoint {} is loaded'.format(AC_MODEL_PATH))

    model.to(device)


    # Import the necessary modules 
    matcher= HungarianMatcher(0.5,1,1,0.5)
    matcher.eval()


    # Clean visualization directory
    import shutil
    output_dir = os.path.join(
        LOG_ROOT,
        "example_visualization",
        CONFIG.dsg["DATASET_NAME"],
        CONFIG.ac["run_name"],
        "visualization_example",
    )
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")


    # Loading the object detector
    object_detector = detector(train=False, object_classes=train_dataset.object_classes, use_SUPPLY=True, model_path=OBJ_DET_MODEL_PATH, mode='sgdet').to(device=device)
    object_detector.eval()

    sttran_conf = STTranConfig()
    dsg_generator = STTran(mode=sttran_conf.mode,
               spatial_class_num=len(train_dataset.spatial_relationships),
               obj_classes=train_dataset.object_classes,
               enc_layer_num=sttran_conf.enc_layer,
               dec_layer_num=sttran_conf.dec_layer).to(device=device)
    dsg_generator.eval()
    ckpt = torch.load(DSG_MODEL_PATH, map_location=device)
    dsg_generator.load_state_dict(ckpt['state_dict'], strict=False)
    print('DSG-Generator checkpoint {} is loaded'.format(DSG_MODEL_PATH))
    print('*'*50)
    # ----------------------------------------------

    # Evaluating model performance
    print('\n'*3)
    print("Evaluating model performance...")

    # Train set        
    train_set_dict = run_epoch(model, matcher, object_detector, dsg_generator,
                                    train_loader, train_dataset, None,
                                    device, None, 0, train=False, visualize_examples=True)
    # Test set
    test_set_dict = run_epoch(model, matcher, object_detector, dsg_generator,
                                    test_loader, test_dataset, None,
                                    device, None, 0, train=False, visualize_examples=True)