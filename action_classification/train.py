from __future__ import annotations

import os
import sys
import datetime
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Patch

import wandb

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

# Add project root (one level up) to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.config_loader import CONFIG

LOG_ROOT = os.environ.get("VISA_LOG_ROOT", "logs")

from lib.net.stgcn import STGCN
from lib.dataloader.dataloader import BlocksworldSequences
from lib.tools.dsg_construction_utils import (
    construct_dynamic_scene_graph_with_tracking,
    construct_dynamic_scene_graph_with_tracking_one_partition,
    gt_to_tracklet_indices,
)
from lib.tools.training_utils import (
    custom_cuda_collate_fn,
    EarlyStopping,
    visualize_predictions,
    obtain_json_annotation,
)
    

from dsg_generator.lib.visualize_predictions import (
    visualize_object_detection,
    visualize_predictions_console,
    visualize_scene_graph,
)
from dsg_generator.lib.track import get_sequence, get_sequence_simple
from dsg_generator.lib.sttran import STTran
from dsg_generator.lib.AdamW import AdamW
from dsg_generator.lib.matcher import HungarianMatcher
from dsg_generator.lib.object_detector import detector
from dsg_generator.lib.config import Config as STTranConfig








def run_epoch(
    model,
    matcher,
    object_detector,
    dsg_generator,
    data_loader,
    dataset,
    optimizer,
    device,
    model_save_dir,
    epoch,
    train=True,
    scheduler=None,
    print_json=False,
    json_name=None,
    visualize_examples=False,
):
    """
    Runs a single training or evaluation epoch for the STGCN-based action classification model.

    This function performs the following steps:
    - Iterates through the dataset in batches.
    - Applies object detection, tracking, and dynamic scene graph (DSG) generation.
    - Constructs node and adjacency matrices for STGCN input.
    - Computes action name and argument prediction losses.
    - Backpropagates gradients (if training), applies optimizer and scheduler.
    - Evaluates accuracy and F1 metrics, optionally visualizes PCA and predictions.
    - Saves model checkpoints after training epochs.

    Returns:
        dict: averaged metrics across the epoch.
    """

    # ------------------------------------------------------------------
    # Config / constants
    # ------------------------------------------------------------------
    sttran_conf = STTranConfig()  # Config for DSG-Generator
    conf = CONFIG.ac

    ACTION_NAME_LOSS_WEIGHT = conf["action_name_loss_weight"]
    ARGUMENT1_LOSS_WEIGHT = conf["argument1_loss_weight"]
    ARGUMENT2_LOSS_WEIGHT = conf["argument2_loss_weight"]
    ARGUMENT_ALL_LOSS_WEIGHT = conf["argument_all_loss_weight"]

    NUM_RELS = len(dataset.spatial_relationships) + 1  # + identity
    NUM_NODES = CONFIG.ac["num_nodes"][CONFIG.dsg["DATASET_NAME"]]
    MAX_ARITY = CONFIG.ac["max_arity"][CONFIG.dsg["DATASET_NAME"]]
    NUM_FEATURES = conf["num_features"]

    PLOT_PCA = conf["show_pca"]
    THRESHOLD_ADD_EDGE = conf["threshold_add_edge"]

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    ACTION_NAME_LOSS = conf["action_name_loss"]
    if ACTION_NAME_LOSS == "CrossEntropyLoss":
        action_name_criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Action name loss function not implemented")

    ARGUMENT_PREDICTION_METHOD = conf["stgcn_model"]["argument_prediction"]["method"]
    ARGUMENT_LOSS = conf["stgcn_model"]["argument_prediction"]["parameters"][ARGUMENT_PREDICTION_METHOD][
        "argument_loss"
    ]

    if ARGUMENT_LOSS == "CrossEntropyLoss":
        argument_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif ARGUMENT_LOSS == "BCEWithLogitsLoss":
        argument_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
    else:
        raise NotImplementedError("Argument loss function not implemented")

    # ------------------------------------------------------------------
    # Train / eval mode
    # ------------------------------------------------------------------
    if train:
        model.train()
    else:
        model.eval()

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    total_loss_combined = 0.0
    total_loss_action_name = 0.0
    total_loss_arg1 = 0.0
    total_loss_arg2 = 0.0
    total_loss_args = 0.0

    total_acc_combined = 0.0
    total_acc_action_name = 0.0
    total_acc_arg1 = 0.0
    total_acc_arg2 = 0.0
    total_acc_args = 0.0

    # PCA collection
    all_pooled_pca_features = []
    all_pooled_pca_labels = []

    all_aggregated_pca_features = []
    all_aggregated_pca_labels = []

    all_aggregated_indexed_pca_features = []
    all_aggregated_indexed_pca_labels = []

    all_aggregated_indexed_room_only_pca_features = []
    all_aggregated_indexed_room_only_pca_labels = []

    all_input_pca_features = []
    all_input_pca_labels = []

    # F1 collection
    all_action_name_preds = []
    all_action_name_targets = []
    all_args_preds = []
    all_args_targets = []

    # Per action type accuracy
    correct_args_per_action_type = [0 for _ in range(len(dataset.action_name_classes) + 1)]
    cnt_args_per_action_type = [0 for _ in range(len(dataset.action_name_classes) + 1)]
    correct_per_action_type = [0 for _ in range(len(dataset.action_name_classes) + 1)]
    cnt_per_action_type = [0 for _ in range(len(dataset.action_name_classes) + 1)]

    json_output = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    skipped_num = 0
    train_iter = iter(data_loader)

    for b in range(len(data_loader)):
        action_names = dataset.action_name_classes
        max_arity = MAX_ARITY

        import copy
        batch = next(train_iter)

        # Batch buffers
        Xs_batch = []
        As_batch = []
        action_name_labels_batch = []
        arguments_gt_tracklet_ids_batch = []

        try:
            for data in batch:
                # ------------------------------------------------------
                # Unpack + clone to device
                # ------------------------------------------------------
                im_data = copy.deepcopy(data[0]).to(device)
                im_info = copy.deepcopy(data[1]).to(device)
                gt_boxes = copy.deepcopy(data[2]).to(device)
                num_boxes = copy.deepcopy(data[3]).to(device)

                dsg_gt_annotation = dataset.dsg_gt_annotations[data[4]]
                action_gt_annotation = dataset.action_annotations[data[4]]
                assert len(dsg_gt_annotation) == len(
                    action_gt_annotation
                ), "Length of dsg_gt_annotation and action_gt_annotation should be the same"

                # ------------------------------------------------------
                # Object detection + tracking + DSG (no grad)
                # ------------------------------------------------------
                with torch.no_grad():
                    entry_object_detector = object_detector(
                        im_data, im_info, gt_boxes, num_boxes, dsg_gt_annotation, im_all=None
                    )

                    get_sequence(
                        entry_object_detector,
                        dsg_gt_annotation,
                        matcher,
                        (im_info[0][:2] / im_info[0, 2]).cpu().data,
                        sttran_conf.mode,
                        GENERAL_COARSE_TRACKING=True,
                    )

                    entry_dsg_generator = dsg_generator(entry_object_detector)

                    get_sequence(
                        entry_dsg_generator,
                        dsg_gt_annotation,
                        matcher,
                        (im_info[0][:2] / im_info[0, 2]).cpu().data,
                        sttran_conf.mode,
                        GENERAL_COARSE_TRACKING=True,
                    )

                # ------------------------------------------------------
                # Optional: train with GT scene graph
                # ------------------------------------------------------
                USE_GT_SCENE_GRAPH_FOR_TRAINING = conf["use_gt_dsg"]
                if USE_GT_SCENE_GRAPH_FOR_TRAINING:
                    print("USING GROUND TRUTH SCENE GRAPH FOR TRAINING")
                    from lib.tools.gt_utils import (
                        frames_to_bbox_tensor,
                        extract_classes,
                        pairs_from_dsg_gt,
                        augment_features_to_2376,
                        populate_image_derived_fields,
                        build_trivial_indices_from_boxes,
                        build_gt_distribution_from_dsg,
                        spatial_distribution_from_dsg_generic,
                    )

                    short = entry_dsg_generator

                    synth_gt = {}
                    synth_gt["boxes"] = frames_to_bbox_tensor(dsg_gt_annotation, device=device)
                    synth_gt["labels"] = extract_classes(dsg_gt_annotation, device=device)
                    synth_gt["scores"] = torch.ones(
                        (synth_gt["boxes"].shape[0],), device=device, dtype=torch.float
                    )
                    synth_gt["pred_labels"] = synth_gt["labels"]
                    synth_gt["im_info"] = short["im_info"]
                    synth_gt["fmaps"] = short["fmaps"]

                    synth_gt["distribution"] = build_gt_distribution_from_dsg(
                        dsg_gt_annotation, object_classes=dataset.object_classes, device=device
                    )
                    synth_gt["pred_scores"] = None

                    pair_idx, im_idx, _ = pairs_from_dsg_gt(
                        dsg_gt_annotation, device=device, include_self_pairs=True
                    )
                    synth_gt["pair_idx"] = pair_idx
                    synth_gt["im_idx"] = im_idx

                    REL_CLASSES = dataset.relationship_classes  # or dataset.spatial_relationships
                    CLASS_GATES = None
                    SYMMETRIC = set()

                    synth_gt["spatial_distribution"] = spatial_distribution_from_dsg_generic(
                        dsg_gt_annotation,
                        pair_idx=synth_gt["pair_idx"],
                        im_idx=synth_gt["im_idx"],
                        rel_classes=REL_CLASSES,
                        device=device,
                        pos_prob=0.999,
                        neg_prob=0.001,
                        unary_on_self=True,
                        class_gates=CLASS_GATES,
                        symmetric_rels=SYMMETRIC,
                    )

                    synth_gt["indices"] = build_trivial_indices_from_boxes(synth_gt["boxes"])

                    populate_image_derived_fields(
                        synth_gt, stride=16, roi_out=(7, 7), force_feat_dim=2048
                    )
                    augment_features_to_2376(
                        synth_gt, object_classes=dataset.object_classes, device=device
                    )

                    entry_dsg_generator = synth_gt

                    get_sequence(
                        entry_dsg_generator,
                        dsg_gt_annotation,
                        matcher,
                        (im_info[0][:2] / im_info[0, 2]).cpu().data,
                        sttran_conf.mode,
                        GENERAL_COARSE_TRACKING=True,
                    )

                # ------------------------------------------------------
                # Tracklet consistency check
                # ------------------------------------------------------
                skip = False
                tracklets = [t for t in entry_dsg_generator["indices"] if t.numel() > 0]
                len_tracklet = min([len(t) for t in tracklets])
                for t in tracklets:
                    if len(t) != len_tracklet:
                        skip = True
                        break
                if skip:
                    skipped_num += 1
                    continue

                # ------------------------------------------------------
                # Feature ablations
                # ------------------------------------------------------
                feature_method = conf["feature_method"]

                if feature_method == "hardcoded":
                    num_classes = len(dataset.object_classes)
                    one_hot_labels = torch.nn.functional.one_hot(
                        entry_dsg_generator["labels"], num_classes=num_classes
                    ).float()

                    new_features = torch.cat(
                        (
                            entry_dsg_generator["boxes"][:, 1:] / 480.0,
                            one_hot_labels,
                            entry_dsg_generator["scores"].unsqueeze(1),
                        ),
                        dim=1,
                    )
                    entry_dsg_generator["features"] = new_features

                elif feature_method == "random_noise":
                    entry_dsg_generator["features"] = torch.randn(
                        entry_dsg_generator["boxes"].shape[0], 100, device=device
                    )

                elif feature_method == "constant":
                    entry_dsg_generator["features"] = torch.ones(
                        entry_dsg_generator["boxes"].shape[0], 100, device=device
                    )

                elif feature_method == "concat_one_hot":
                    if "features" not in entry_dsg_generator:
                        raise KeyError(
                            "entry_dsg_generator has no 'features' to concatenate with one-hot labels"
                        )

                    if "labels" in entry_dsg_generator:
                        labels = entry_dsg_generator["labels"]
                    elif "pred_labels" in entry_dsg_generator:
                        labels = entry_dsg_generator["pred_labels"]
                    else:
                        raise KeyError(
                            "No 'labels' or 'pred_labels' found in entry_dsg_generator for one-hot encoding"
                        )

                    if not torch.is_tensor(labels):
                        labels = torch.tensor(labels, device=device)

                    labels = labels.long().view(-1)
                    num_classes = len(dataset.object_classes)
                    one_hot = torch.nn.functional.one_hot(
                        labels, num_classes=num_classes
                    ).float().to(entry_dsg_generator["features"].device)

                    entry_dsg_generator["features"] = torch.cat(
                        (entry_dsg_generator["features"], one_hot), dim=1
                    )

                elif feature_method == "original":
                    pass

                # ------------------------------------------------------
                # Construct DSG (node features + adjacency)
                # ------------------------------------------------------
                if CONFIG.ac["ablation_studies"]["use_simple_edge_type"]:
                    Xs, As = construct_dynamic_scene_graph_with_tracking_one_partition(
                        entry_dsg_generator,
                        thresh=THRESHOLD_ADD_EDGE,
                        padding=NUM_NODES,
                        num_rels=NUM_RELS,
                    )
                    Xs_labels = None  # may not be available in this mode
                else:
                    Xs, Xs_labels, As = construct_dynamic_scene_graph_with_tracking(
                        entry_dsg_generator,
                        thresh=THRESHOLD_ADD_EDGE,
                        padding=NUM_NODES,
                        num_rels=NUM_RELS,
                    )

                # ------------------------------------------------------
                # Debug adjacency
                # ------------------------------------------------------
                DEBUG_ADJACENCY = False
                if DEBUG_ADJACENCY:
                    rel_classes = dataset.relationship_classes + ["identity"]
                    for i in range(As.shape[1]):
                        from lib.tools.training_utils import visualize_adj_with_labels
                        visualize_adj_with_labels(
                            A=As[0, i],
                            labels=Xs_labels[0],
                            class_names=dataset.object_classes,
                            title=f"Adjacency matrix for relation {rel_classes[i]}",
                            save_path=f"debug_adj_rel{i}.png",
                            show=False,
                        )
                    import pdb
                    pdb.set_trace()

                # ------------------------------------------------------
                # Debug features
                # ------------------------------------------------------
                DEBUG_FEATURES = False
                if DEBUG_FEATURES:

                    def a(tensor):
                        """Return True if the entire tensor is zero, else False."""
                        return torch.all(tensor == 0).item()

                    for i in range(As.shape[0]):
                        for j in range(As.shape[1]):
                            print(f"As[{i},{j}] all zero: {a(As[i, j])}")

                    import pdb
                    pdb.set_trace()

                # ------------------------------------------------------
                # GT action labels
                # ------------------------------------------------------
                action_name_labels = [action_names.index(frame["predicate"]) for frame in action_gt_annotation]
                action_name_labels = action_name_labels[1:]  # drop first frame
                action_name_labels = torch.tensor(action_name_labels, dtype=torch.long)
                action_name_labels_batch.append(action_name_labels)

                # ------------------------------------------------------
                # GT argument labels -> tracklet indices
                # ------------------------------------------------------
                action_names_gt_ids = [action_names.index(frame["predicate"]) for frame in action_gt_annotation]
                matched_indices = gt_to_tracklet_indices(
                    dsg_gt_annotation, entry_dsg_generator["boxes"], tracklets
                )

                arguments_gt_identifiers = []
                for i in range(max_arity):
                    identifiers = [
                        None if len(frame["args"]) <= i else frame["args"][i]["identifier"]
                        for frame in action_gt_annotation
                    ]
                    arguments_gt_identifiers.append(identifiers)

                if max_arity > 0:
                    arguments_gt_tracklet_ids = copy.deepcopy(arguments_gt_identifiers)
                    arguments_gt_tracklet_ids = [args[1:] for args in arguments_gt_tracklet_ids]

                    for i in range(len(arguments_gt_tracklet_ids[0])):
                        for arity in range(max_arity):
                            if arguments_gt_tracklet_ids[arity][i] is None:
                                arguments_gt_tracklet_ids[arity][i] = arguments_gt_tracklet_ids[0][i]
                            else:
                                arguments_gt_tracklet_ids[arity][i] = matched_indices[i][
                                    arguments_gt_tracklet_ids[arity][i]
                                ]

                    for l_i, lst in enumerate(arguments_gt_tracklet_ids):
                        for e_i, ele in enumerate(lst):
                            if ele is None:
                                arguments_gt_tracklet_ids[l_i][e_i] = -1

                    arguments_gt_tracklet_ids = torch.tensor(
                        arguments_gt_tracklet_ids, dtype=torch.long
                    )
                    arguments_gt_tracklet_ids_batch.append(arguments_gt_tracklet_ids)

                # ------------------------------------------------------
                # Shape checks + add to batch
                # ------------------------------------------------------
                assert Xs.shape[1:] == (NUM_NODES, NUM_FEATURES), (
                    f"Xs shape is {Xs.shape} instead of {(NUM_NODES, NUM_FEATURES)}"
                )
                assert As.shape[1:] == (
                    len(dataset.spatial_relationships) + 1,
                    NUM_NODES,
                    NUM_NODES,
                ), (
                    f"As shape is {As.shape} instead of "
                    f"{(len(dataset.spatial_relationships) + 1, NUM_NODES, NUM_NODES)}"
                )

                Xs_batch.append(Xs)
                As_batch.append(As)

            # ----------------------------------------------------------
            # Stack batch
            # ----------------------------------------------------------
            Xs = torch.stack(Xs_batch, dim=0).to(device)
            As = torch.stack(As_batch, dim=0).to(device)
            action_name_labels = torch.stack(action_name_labels_batch, dim=0).to(device)

            if max_arity > 0:
                arguments_gt_tracklet_ids = torch.stack(arguments_gt_tracklet_ids_batch, dim=0).to(device)
                arguments_gt_tracklet_ids = arguments_gt_tracklet_ids.permute(1, 0, 2)
            else:
                arguments_gt_tracklet_ids = (
                    torch.zeros((1, Xs.shape[0], Xs.shape[1]), dtype=torch.long).to(device) - 1
                )

            if action_name_labels.numel() == 0:
                print("No action name labels found. Skipping this sample.")
                skipped_num += 1
                continue

            # ----------------------------------------------------------
            # Forward pass
            # ----------------------------------------------------------
            pred_action_name, pred_arguments, _debug_dict = model(Xs, As)

            # ----------------------------------------------------------
            # Argument labels formatting for loss
            # ----------------------------------------------------------
            if ARGUMENT_LOSS == "CrossEntropyLoss":
                arg1_labels = arguments_gt_tracklet_ids[0]
                arg2_labels = arguments_gt_tracklet_ids[1]

            elif ARGUMENT_LOSS == "BCEWithLogitsLoss":
                args_labels = torch.zeros(
                    arguments_gt_tracklet_ids.shape[1], Xs.shape[1] - 1, NUM_NODES
                ).to(device)

                for b_id in range(arguments_gt_tracklet_ids.shape[1]):
                    for index in range(arguments_gt_tracklet_ids.shape[2]):
                        for arity in range(max_arity):
                            try:
                                if arguments_gt_tracklet_ids[arity, b_id, index] != -1:
                                    args_labels[b_id, index, arguments_gt_tracklet_ids[arity, b_id, index]] = 1
                            except:
                                import pdb

                                pdb.set_trace()

            # ----------------------------------------------------------
            # Visualize examples
            # ----------------------------------------------------------
            if visualize_examples:
                print(f"Visualizing example {b+1}/{len(data_loader)}")
                try:
                    visualize_predictions(
                        im_info,
                        im_data,
                        entry_dsg_generator,
                        dsg_gt_annotation,
                        action_gt_annotation,
                        dataset,
                        b,
                        ARGUMENT_LOSS,
                        pred_action_name[-1],
                        pred_arguments[0],
                        action_names,
                        matcher,
                        sttran_conf,
                        action_name_labels[-1],
                        arguments_gt_tracklet_ids[:, -1] + 1,
                        max_arity,
                        gt_boxes,
                    )
                except Exception as e:
                    print(f"Error during visualization: {e}")
                    import traceback

                    traceback.print_exc()

            # ----------------------------------------------------------
            # JSON export
            # ----------------------------------------------------------
            if print_json:
                ret = obtain_json_annotation(
                    im_info,
                    im_data,
                    entry_dsg_generator,
                    dsg_gt_annotation,
                    action_gt_annotation,
                    dataset,
                    b,
                    ARGUMENT_LOSS,
                    pred_action_name[-1],
                    [lst[-1] for lst in pred_arguments],
                    action_names,
                    matcher,
                    sttran_conf,
                    action_name_labels[-1],
                    arguments_gt_tracklet_ids[:, -1] + 1,
                    max_arity,
                )
                json_output.extend(ret)

            # ----------------------------------------------------------
            # PCA collection
            # ----------------------------------------------------------
            if PLOT_PCA:
                pooled_features = _debug_dict.get("x_pooled_paired", None)
                if pooled_features is not None and pooled_features.shape[0] > 0:
                    try:
                        B, Tm1, D = pooled_features.shape
                        features = pooled_features.detach().cpu().numpy().reshape(-1, D)
                        labels = action_name_labels.detach().cpu().numpy().reshape(-1)
                        all_pooled_pca_features.append(features)
                        all_pooled_pca_labels.append(labels)
                    except Exception as e:
                        print(f"[PCA collection error, 1] {e}")

                # Handle Xs_labels shape (if available)
                if Xs_labels is not None:
                    if not torch.is_tensor(Xs_labels):
                        Xs_labels = torch.tensor(Xs_labels, dtype=torch.long)

                    if Xs_labels.dim() == 2:
                        Xs_labels = Xs_labels.unsqueeze(0).expand(B, -1, -1)
                    elif Xs_labels.dim() == 3:
                        pass
                    else:
                        raise ValueError(f"Xs_labels must be [T,N] or [B,T,N], got {Xs_labels.shape}")

                # 2) aggregated features
                aggregated_features = _debug_dict.get("x_aggregated_features", None)
                if aggregated_features is not None and aggregated_features.shape[0] > 0:
                    try:
                        assert aggregated_features.dim() == 4, f"expected [B,F,T,N], got {aggregated_features.shape}"
                        B, F, T, N = aggregated_features.shape

                        feats = aggregated_features.permute(0, 2, 3, 1).contiguous()
                        feats = feats.view(B * T * N, F).detach().cpu().numpy()

                        labs = Xs_labels.reshape(B * T * N).detach().cpu().numpy()

                        all_aggregated_pca_features.append(feats)
                        all_aggregated_pca_labels.append(labs)
                    except Exception as e:
                        print(f"[PCA collection error, 2] {e}")

                # 2b) aggregated indexed features
                aggregated_features = _debug_dict.get("x_aggregated_features", None)
                if aggregated_features is not None and aggregated_features.shape[0] > 0:
                    try:
                        assert aggregated_features.dim() == 4, f"expected [B,F,T,N], got {aggregated_features.shape}"
                        B, F, T, N = aggregated_features.shape

                        feats = aggregated_features.permute(0, 2, 3, 1).contiguous()
                        feats = feats.view(B * T * N, F).detach().cpu().numpy()

                        seq = torch.arange(1, N + 1)
                        seq = seq.view(1, 1, N).expand(B, T, N)

                        Xs_labels_indexed = [
                            [
                                [
                                    (Xs_labels[b, t, n].item(), seq[b, t, n].item())
                                    for n in range(Xs_labels.size(2))
                                ]
                                for t in range(Xs_labels.size(1))
                            ]
                            for b in range(Xs_labels.size(0))
                        ]
                        Xs_labels_indexed = torch.tensor(Xs_labels_indexed, dtype=torch.long, device=device)

                        labs = Xs_labels_indexed.reshape(B * T * N, 2).detach().cpu().numpy()

                        all_aggregated_indexed_pca_features.append(feats)
                        all_aggregated_indexed_pca_labels.append(labs)
                    except Exception as e:
                        print(f"[PCA collection error, 2] {e}")

                # 2c) indexed room only (disabled in your original)
                aggregated_features = _debug_dict.get("x_aggregated_features", None)
                if aggregated_features is not None and aggregated_features.shape[0] > 0 and False:
                    try:
                        assert aggregated_features.dim() == 4, f"expected [B,F,T,N], got {aggregated_features.shape}"
                        B, F, T, N = aggregated_features.shape

                        feats = aggregated_features.permute(0, 2, 3, 1).contiguous()
                        feats = feats.view(B * T * N, F).detach().cpu().numpy()

                        seq = torch.arange(1, N + 1)
                        seq = seq.view(1, 1, N).expand(B, T, N)

                        Xs_labels_indexed = [
                            [
                                [
                                    (Xs_labels[b, t, n].item(), seq[b, t, n].item())
                                    for n in range(Xs_labels.size(2))
                                ]
                                for t in range(Xs_labels.size(1))
                            ]
                            for b in range(Xs_labels.size(0))
                        ]
                        Xs_labels_indexed = torch.tensor(Xs_labels_indexed, dtype=torch.long, device=device)

                        labs = Xs_labels_indexed.reshape(B * T * N, 2).detach().cpu().numpy()

                        room_class_id = dataset.object_classes.index("room")
                        room_indices = [i for i, lab in enumerate(labs) if lab[0] == room_class_id]
                        feats = feats[room_indices]
                        labs = labs[room_indices]

                        all_aggregated_indexed_room_only_pca_features.append(feats)
                        all_aggregated_indexed_room_only_pca_labels.append(labs)

                    except Exception as e:
                        print(f"[PCA collection error, 2] {e}")

                # 3) input features
                input_features = _debug_dict.get("x_input_features", None)
                if input_features is not None and input_features.shape[0] > 0:
                    try:
                        assert input_features.dim() == 4, f"expected [B,F,T,N], got {input_features.shape}"
                        B, F, T, N = input_features.shape

                        feats = input_features.permute(0, 2, 3, 1).contiguous()
                        feats = feats.view(B * T * N, F).detach().cpu().numpy()

                        labs = Xs_labels.reshape(B * T * N).detach().cpu().numpy()

                        all_input_pca_features.append(feats)
                        all_input_pca_labels.append(labs)
                    except Exception as e:
                        print(f"[PCA collection error, 3] {e}")

            # ----------------------------------------------------------
            # Loss computation
            # ----------------------------------------------------------
            pred_action_name = pred_action_name.permute(0, 2, 1)
            action_name_loss = action_name_criterion(pred_action_name, action_name_labels)

            if ARGUMENT_LOSS == "CrossEntropyLoss":
                pred_arguments[0] = pred_arguments[0].permute(0, 2, 1)
                pred_arguments[1] = pred_arguments[1].permute(0, 2, 1)

                arg1_loss = argument_criterion(pred_arguments[0], arg1_labels)
                arg2_loss = argument_criterion(pred_arguments[1], arg2_labels)
                args_loss = torch.tensor(0.0).to(device)

            elif ARGUMENT_LOSS == "BCEWithLogitsLoss":
                arg1_loss = torch.tensor(0.0).to(device)
                arg2_loss = torch.tensor(0.0).to(device)
                args_loss = argument_criterion(pred_arguments[0], args_labels.float())

            combined_loss = (
                action_name_loss * ACTION_NAME_LOSS_WEIGHT
                + arg1_loss * ARGUMENT1_LOSS_WEIGHT
                + arg2_loss * ARGUMENT2_LOSS_WEIGHT
                + args_loss * ARGUMENT_ALL_LOSS_WEIGHT
            )

            total_loss_combined += combined_loss.item()
            total_loss_action_name += action_name_loss.item()
            total_loss_arg1 += arg1_loss.item()
            total_loss_arg2 += arg2_loss.item()
            total_loss_args += args_loss.item()

            # ----------------------------------------------------------
            # Backprop
            # ----------------------------------------------------------
            if train:
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(combined_loss)

            # ----------------------------------------------------------
            # Accuracy
            # ----------------------------------------------------------
            predicted_action_name = torch.max(torch.softmax(pred_action_name, dim=1), dim=1).indices
            correct_action_names = (predicted_action_name == action_name_labels).sum().item()
            acc_action_name = correct_action_names / (action_name_labels.shape[0] * action_name_labels.shape[1])

            # per action type acc
            for i in range(len(action_name_labels[0])):
                gt_action = action_name_labels[0][i].item()
                hat_action = predicted_action_name[0][i].item()
                if gt_action == hat_action:
                    correct_per_action_type[gt_action] += 1
                cnt_per_action_type[gt_action] += 1

            if ARGUMENT_LOSS == "CrossEntropyLoss":
                predicted_arg1 = torch.max(torch.softmax(pred_arguments[0], dim=1), dim=1).indices
                correct_arg1 = (predicted_arg1 == arg1_labels).sum().item()
                acc_arg1 = correct_arg1 / (arg1_labels.shape[0] * arg1_labels.shape[1])

                predicted_arg2 = torch.max(torch.softmax(pred_arguments[1], dim=1), dim=1).indices
                correct_arg2 = (predicted_arg2 == arg2_labels).sum().item()
                acc_arg2 = correct_arg2 / (arg2_labels.shape[0] * arg2_labels.shape[1])

                acc_args = 0
                combined_acc = (acc_arg1 + acc_arg2 + acc_action_name) / 3

            elif ARGUMENT_LOSS == "BCEWithLogitsLoss":
                acc_arg1 = 0
                acc_arg2 = 0

                thresh = 0.5
                correct_args = ((torch.sigmoid(pred_arguments[0]) > thresh) == args_labels).all(dim=-1).sum().item()
                acc_args = correct_args / (args_labels.shape[0] * args_labels.shape[1])

                combined_acc = (acc_args + acc_action_name) / 2

                # F1: arguments
                all_args_preds.append((torch.sigmoid(pred_arguments[0]) > 0.5).cpu().int().view(-1, NUM_NODES))
                all_args_targets.append(args_labels.cpu().int().view(-1, NUM_NODES))

                # per-action-type argument acc
                corrs = ((torch.sigmoid(pred_arguments[0]) > thresh) == args_labels)[0]
                for i in range(len(action_name_labels[0])):
                    action_class = action_name_labels[0][i].item()
                    if corrs[i].all():
                        correct_args_per_action_type[action_class] += 1
                    cnt_args_per_action_type[action_class] += 1

            # F1: action type
            all_action_name_preds.append(predicted_action_name.cpu().flatten())
            all_action_name_targets.append(action_name_labels.cpu().flatten())

            total_acc_combined += combined_acc
            total_acc_action_name += acc_action_name
            total_acc_arg1 += acc_arg1
            total_acc_arg2 += acc_arg2
            total_acc_args += acc_args

            # ----------------------------------------------------------
            # W&B logging
            # ----------------------------------------------------------
            if train and wandb.run is not None:
                wandb.log(
                    {
                        "batch/loss_combined": combined_loss.item(),
                        "batch/loss_action_name": action_name_loss.item(),
                        "batch/loss_args": args_loss.item() if ARGUMENT_LOSS == "BCEWithLogitsLoss" else 0.0,
                        "batch/acc_action_name": acc_action_name,
                        "batch/acc_args": acc_args,
                        "batch/epoch": epoch,
                        "batch/index": b,
                    }
                )

            # ----------------------------------------------------------
            # Console logging
            # ----------------------------------------------------------
            if train:
                if ARGUMENT_LOSS == "CrossEntropyLoss":
                    VERBOSE_OUTPUT = True
                    if VERBOSE_OUTPUT:
                        print(
                            f"e {epoch}, b {b+1}/{len(data_loader)}, "
                            f"l_act_name = {action_name_loss.item():.4f}, "
                            f"l_a1 = {arg1_loss.item():.4f}, "
                            f"l_a2 = {arg2_loss.item():.4f}, "
                            f"a_act_name = {acc_action_name:.4f}, "
                            f"a_a1 = {acc_arg1:.4f}, "
                            f"a_a2 = {acc_arg2:.4f}"
                        )
                    else:
                        print(
                            f"e {epoch}, b {b+1}/{len(data_loader)}, "
                            f"loss_combined = {combined_loss.item():.4f}, "
                            f"accuracy_combined = {combined_acc:.4f} "
                        )

                elif ARGUMENT_LOSS == "BCEWithLogitsLoss":
                    print(
                        f"e {epoch}, b {b+1}/{len(data_loader)}, "
                        f"l_act_name = {action_name_loss.item():.4f}, "
                        f"l_args = {args_loss.item():.4f}, "
                        f"a_act_name = {acc_action_name:.4f}, "
                        f"a_args = {acc_args:.4f} "
                    )

        except Exception as e:
            print(f"Error processing batch {b+1} in epoch {epoch}: {e}")
            import traceback

            traceback.print_exc()
            skipped_num += 1
            continue

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    if train:
        print(f"Saving model after epoch {epoch}")
        import os

        model_path = os.path.join(model_save_dir, f"model_{epoch}.tar")
        torch.save({"state_dict": model.state_dict()}, model_path)

        best_model_path = os.path.join(model_save_dir, "model_best.tar")
        torch.save({"state_dict": model.state_dict()}, best_model_path)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if print_json:
        from lib.tools.dsg_construction_utils import dumps_compact_lists
        import os

        folder = os.path.join(
            LOG_ROOT,
            "ac_results",
            CONFIG.dsg["DATASET_NAME"],
            CONFIG.ac["run_name"],
            "json",
        )
        os.makedirs(folder, exist_ok=True)
        output_file = os.path.join(folder, f"{json_name}")

        s = dumps_compact_lists(json_output, indent=4, max_inline_len=8, max_width=100)
        with open(output_file, "w") as f:
            f.write(s)

    # ------------------------------------------------------------------
    # PCA plotting at end of epoch
    # ------------------------------------------------------------------
    if PLOT_PCA and (
        len(all_pooled_pca_features) > 0
        or len(all_aggregated_pca_features) > 0
        or len(all_input_pca_features) > 0
    ):
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import numpy as np
        import os
        import datetime
        from lib.tools.training_utils import _export_projector_tsv, _pca_and_scatter


        try:
            from lib.tools.training_utils import _pca_and_scatter_indexed

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_root = os.path.join(
                LOG_ROOT,
                "ac_results",
                CONFIG.dsg["DATASET_NAME"],
                CONFIG.ac["run_name"],
            )
            pca_output_dir = os.path.join(
                results_root,
                "feature_visualization",
                "pca",
                dataset.mode,
                timestamp,
            )
            npz_output_dir = os.path.join(
                results_root,
                "feature_visualization",
                "npz",
                dataset.mode,
                timestamp,
            )
            os.makedirs(pca_output_dir, exist_ok=True)
            os.makedirs(npz_output_dir, exist_ok=True)

            def save_npz(out_dir, stem, **arrays):
                path = os.path.join(out_dir, f"{stem}.npz")
                np.savez_compressed(path, **arrays)

            # 1) pooled features (action class)
            if len(all_pooled_pca_features) > 0:
                X = np.concatenate(all_pooled_pca_features, axis=0)
                y = np.concatenate(all_pooled_pca_labels, axis=0)
                img_mapping = [i for i, img in enumerate(all_pooled_pca_labels) for _ in range(len(img))]

                action_names = dataset.action_name_classes if hasattr(dataset, "action_name_classes") else None

                _pca_and_scatter(
                    X,
                    y,
                    title="PCA of Pooled Features (Action Class)",
                    label_names=action_names,
                    out_dir=pca_output_dir,
                    stem=f"pca_pooled_epoch_{epoch}",
                )

                save_npz(
                    npz_output_dir,
                    stem=f"pooled_epoch_{epoch}",
                    X=X,
                    y=y,
                    img_mapping=img_mapping,
                    action_names=action_names,
                )

            # 2) aggregated features (object class)
            if len(all_aggregated_pca_features) > 0:
                X = np.concatenate(all_aggregated_pca_features, axis=0)
                y = np.concatenate(all_aggregated_pca_labels, axis=0)
                img_mapping = [i for i, img in enumerate(all_aggregated_pca_labels) for _ in range(len(img))]

                obj_names = dataset.object_classes if hasattr(dataset, "object_classes") else None

                _pca_and_scatter(
                    X,
                    y,
                    title="PCA of Node Aggregated Features (Object Class)",
                    label_names=obj_names,
                    out_dir=pca_output_dir,
                    stem=f"pca_nodes_epoch_{epoch}",
                )

                save_npz(
                    npz_output_dir,
                    stem=f"aggregated_epoch_{epoch}",
                    X=X,
                    y=y,
                    img_mapping=img_mapping,
                    object_names=obj_names,
                )

            # 2b) aggregated indexed
            if len(all_aggregated_indexed_pca_features) > 0:
                X = np.concatenate(all_aggregated_indexed_pca_features, axis=0)
                y = np.concatenate(all_aggregated_indexed_pca_labels, axis=0)

                _pca_and_scatter_indexed(
                    X,
                    y,
                    title="PCA of Node Aggregated Features (ObjClass/Index)",
                    class_names=dataset.object_classes,
                    out_dir=pca_output_dir,
                    stem=f"pca_nodes_indexed_epoch_{epoch}",
                )

            # 2c) room only
            if len(all_aggregated_indexed_room_only_pca_features) > 0:
                X = np.concatenate(all_aggregated_indexed_room_only_pca_features, axis=0)
                y = np.concatenate(all_aggregated_indexed_room_only_pca_labels, axis=0)

                _pca_and_scatter_indexed(
                    X,
                    y,
                    title="PCA of Node Aggregated Features (ObjClass/Index)",
                    class_names=dataset.object_classes,
                    out_dir=pca_output_dir,
                    stem=f"pca_nodes_indexed_room_only_epoch_{epoch}",
                )

            # 3) input features
            if len(all_input_pca_features) > 0:
                X = np.concatenate(all_input_pca_features, axis=0)
                y = np.concatenate(all_input_pca_labels, axis=0)
                img_mapping = [i for i, img in enumerate(all_input_pca_labels) for _ in range(len(img))]

                obj_names = dataset.object_classes if hasattr(dataset, "object_classes") else None

                _pca_and_scatter(
                    X,
                    y,
                    title="PCA of Raw/Input Node Features",
                    label_names=obj_names,
                    out_dir=pca_output_dir,
                    stem=f"pca_input_epoch_{epoch}",
                )

                save_npz(
                    npz_output_dir,
                    stem=f"input_epoch_{epoch}",
                    X=X,
                    y=y,
                    img_mapping=img_mapping,
                    object_names=obj_names,
                )

        except Exception as e:
            print(f"[PCA Plot Error] {e}")

    # ------------------------------------------------------------------
    # F1 computation
    # ------------------------------------------------------------------
    import numpy as np
    from sklearn.metrics import f1_score

    y_true_action = torch.cat(all_action_name_targets).numpy()
    y_pred_action = torch.cat(all_action_name_preds).numpy()
    f1_action_macro = f1_score(y_true_action, y_pred_action, average="macro", zero_division=0)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    action_names = dataset.action_name_classes
    present_labels = np.unique(np.concatenate((y_true_action, y_pred_action)))

    label_mapping = {
        "pickup_lc": "Pickup L",
        "pickup_rc": "Pickup R",
        "putdown_lc": "Putdown L",
        "putdown_rc": "Putdown R",
        "rotate_vertically": "Rotate V",
        "rotate_horizontally": "Rotate H",
        "stack_horizontally_l_rh": "Stack H1",
        "stack_horizontally_l_r": "Stack H2",
        "stack_vertically_l_r": "Stack V1",
        "unstack_vertically_l_r": "Unstack V",
        "unstack_horizontally_l_rh": "Unstack H1",
        "unstack_horizontally_lh_r": "Unstack H2",
        "unstack_horizontally_l_r": "Unstack H3",
    }

    present_action_names = [label_mapping.get(action_names[i], action_names[i]) for i in present_labels]
    cm_present = confusion_matrix(y_true_action, y_pred_action, labels=present_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_present, display_labels=present_action_names)
    disp.plot(xticks_rotation=45, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix: Action Type Prediction")
    plt.tight_layout()

    output_dir = os.path.join(
        LOG_ROOT,
        "ac_results",
        CONFIG.dsg["DATASET_NAME"],
        CONFIG.ac["run_name"],
        "confusion_matrix",
    )
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{dataset.mode}_set_epoch_{epoch}.pdf"))
    plt.close()

    if ARGUMENT_LOSS == "BCEWithLogitsLoss":
        y_true_args = torch.cat(all_args_targets).numpy()
        y_pred_args = torch.cat(all_args_preds).numpy()
        f1_args_macro = f1_score(y_true_args, y_pred_args, average="macro", zero_division=0)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    res_dict = {}
    num_total = len(data_loader) - skipped_num
    if num_total == 0:
        raise ValueError("No valid samples in the data loader. Check your dataset and filtering criteria.")

    # Loss
    res_dict["loss_combined"] = total_loss_combined / num_total
    res_dict["loss_action_name"] = total_loss_action_name / num_total
    res_dict["loss_arg1"] = total_loss_arg1 / num_total
    res_dict["loss_arg2"] = total_loss_arg2 / num_total
    res_dict["loss_args"] = total_loss_args / num_total

    # Accuracy
    res_dict["acc_combined"] = total_acc_combined / num_total
    res_dict["acc_action_name"] = total_acc_action_name / num_total
    res_dict["acc_arg1"] = total_acc_arg1 / num_total
    res_dict["acc_arg2"] = total_acc_arg2 / num_total
    res_dict["acc_args"] = total_acc_args / num_total

    # F1
    res_dict["f1_action_name_macro"] = f1_action_macro
    if ARGUMENT_LOSS == "BCEWithLogitsLoss":
        res_dict["f1_args_macro"] = f1_args_macro

    # Per action type argument accuracy
    args_cnt = 0
    args_sum = 0.0
    for action_type in range(1, len(cnt_args_per_action_type)):
        if cnt_args_per_action_type[action_type] == 0:
            res_dict[f"args_acc_{action_type}"] = -1
        else:
            res_dict[f"args_acc_{action_type}"] = (
                correct_args_per_action_type[action_type] / cnt_args_per_action_type[action_type]
            )
            args_cnt += 1
            args_sum += res_dict[f"args_acc_{action_type}"]
    res_dict["args_acc_mean"] = (args_sum / args_cnt) if args_cnt > 0 else -1

    type_cnt = 0
    type_sum = 0.0
    for action_type in range(1, len(cnt_args_per_action_type)):
        if cnt_per_action_type[action_type] == 0:
            res_dict[f"type_acc_{action_type}"] = -1
        else:
            res_dict[f"type_acc_{action_type}"] = correct_per_action_type[action_type] / cnt_per_action_type[action_type]
            type_cnt += 1
            type_sum += res_dict[f"type_acc_{action_type}"]
    res_dict["type_acc_mean"] = (type_sum / type_cnt) if type_cnt > 0 else -1

    return res_dict








def run_training():
    """
    Trains and evaluates the STGCN model for action classification and argument prediction using dynamic scene graphs.

    The training loop handles:
    - Loading dataset, model, object detector, and DSG generator.
    - Running training and evaluation epochs.
    - Computing and logging losses and accuracies.
    - Optionally applying learning rate scheduling and early stopping.
    - Saving models and optionally visualizing PCA and confusion matrices.

    Returns:
        Tuple[dict, dict]: (train_set_dict, test_set_dict) from the final epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    conf = CONFIG.ac
    dsg_conf = CONFIG.dsg

    num_epochs = conf["num_epochs"]
    batch_size = conf["batch_size"]
    temporal_kernel_size = conf["temporal_kernel_size"]

    num_nodes = CONFIG.ac["num_nodes"][CONFIG.dsg["DATASET_NAME"]]
    num_features = conf["num_features"]

    arg_method = conf["stgcn_model"]["argument_prediction"]["method"]
    optimizer_name = conf["stgcn_model"]["argument_prediction"]["parameters"][arg_method]["optimizer"]

    use_lr_scheduler = conf["use_lr_scheduler"]
    use_early_stopping = conf["use_early_stopping"]
    early_stopping_patience = conf["early_stopping_patience"]
    early_stopping_min_delta = conf["early_stopping_min_delta"]

    print(f"Optimizer: {optimizer_name}")

    # Optimizer hyperparameters (kept identical behavior; just cleaner)
    if optimizer_name == "Adam":
        adam_lr = conf["optimizer_params"]["Adam"]["learning_rate"]
        adam_wd = conf["optimizer_params"]["Adam"]["weight_decay"]
    elif optimizer_name == "AdamW":
        adamw_lr = conf["optimizer_params"]["AdamW"]["learning_rate"]
        adamw_wd = conf["optimizer_params"]["AdamW"]["weight_decay"]
    elif optimizer_name == "SGD":
        sgd_lr = conf["optimizer_params"]["SGD"]["learning_rate"]
        sgd_momentum = conf["optimizer_params"]["SGD"]["momentum"]
        sgd_wd = conf["optimizer_params"]["SGD"]["weight_decay"]
    elif optimizer_name == "RMSprop":
        rmsprop_lr = conf["optimizer_params"]["RMSprop"]["learning_rate"]
        rmsprop_alpha = conf["optimizer_params"]["RMSprop"]["alpha"]
        rmsprop_wd = conf["optimizer_params"]["RMSprop"]["weight_decay"]

    # Paths
    dataset_dir = dsg_conf["data_path"]
    obj_det_model_path = dsg_conf["detector_model_path"]
    dsg_model_path = dsg_conf["dsg_model_path"]

    # ------------------------------------------------------------------
    # Datasets / loaders
    # ------------------------------------------------------------------
    train_dataset = BlocksworldSequences(mode="train", root_dir=dataset_dir)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=custom_cuda_collate_fn,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataset = BlocksworldSequences(mode="test", root_dir=dataset_dir)
    test_loader = DataLoader(
        test_dataset,
        collate_fn=custom_cuda_collate_fn,
        batch_size=batch_size,
        shuffle=False,
    )

    # Spatial kernel size via number of relationships (+1 self-loop)
    spatial_kernel_size = len(train_dataset.spatial_relationships) + 1

    # ------------------------------------------------------------------
    # W&B init
    # ------------------------------------------------------------------
    import time

    wandb.init(
        project="ac_dsg",
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "spatial_kernel_size": spatial_kernel_size,
            "temporal_kernel_size": temporal_kernel_size,
            "num_nodes": num_nodes,
            "num_features": num_features,
            "optimizer": optimizer_name,
            "use_lr_scheduler": use_lr_scheduler,
            "use_early_stopping": use_early_stopping,
            "dataset": CONFIG.dsg["DATASET_NAME"],
            "run_name": CONFIG.ac["run_name"],
        },
        name=f"{CONFIG.dsg['DATASET_NAME']}_{CONFIG.ac['run_name']}_{time.strftime('%Y%m%d_%H%M%S')}" or None,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = STGCN(
        num_classes=train_dataset.num_classes,
        in_channels=num_features,
        spatial_kernel_size=spatial_kernel_size,
        temporal_kernel_size=temporal_kernel_size,
        num_nodes=num_nodes,
    )

    # W&B: watch model
    wandb.watch(model, log="all", log_freq=100)

    # Optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=adam_wd)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=adamw_lr, weight_decay=adamw_wd)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=sgd_lr, momentum=sgd_momentum, weight_decay=sgd_wd
        )
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=rmsprop_lr, alpha=rmsprop_alpha, weight_decay=rmsprop_wd
        )
    else:
        raise NotImplementedError("Optimizer not implemented")

    # Scheduler
    if use_lr_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)

    # Early stopping
    if use_early_stopping:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience, min_delta=early_stopping_min_delta
        )

    model.to(device)

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    run_name = CONFIG.ac["run_name"] + "/" if CONFIG.ac["run_name"] != "" else ""
    model_save_dir = f"models/action_classification/{dataset_dir.split('/')[-1]}/{run_name}"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # ------------------------------------------------------------------
    # Matcher, detector, DSG generator
    # ------------------------------------------------------------------
    matcher = HungarianMatcher(0.5, 1, 1, 0.5)
    matcher.eval()

    object_detector = detector(
        train=False,
        object_classes=train_dataset.object_classes,
        use_SUPPLY=True,
        model_path=obj_det_model_path,
        mode="sgdet",
    ).to(device=device)
    object_detector.eval()

    sttran_conf = STTranConfig()
    dsg_generator = STTran(
        mode=sttran_conf.mode,
        spatial_class_num=len(train_dataset.spatial_relationships),
        obj_classes=train_dataset.object_classes,
        enc_layer_num=sttran_conf.enc_layer,
        dec_layer_num=sttran_conf.dec_layer,
    ).to(device=device)
    dsg_generator.eval()

    ckpt = torch.load(dsg_model_path, map_location=device)
    dsg_generator.load_state_dict(ckpt["state_dict"], strict=False)

    print()
    print("*" * 50)
    print(f"DSG-Generator checkpoint {dsg_model_path} is loaded")
    print("*" * 50)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    import time

    for epoch in range(num_epochs):
        print()
        print(f"Epoch {epoch}/{num_epochs}")
        start_time = time.time()

        # Train one epoch
        _ = run_epoch(
            model,
            matcher,
            object_detector,
            dsg_generator,
            train_loader,
            train_dataset,
            optimizer,
            device,
            model_save_dir,
            epoch,
            train=True,
            scheduler=scheduler if use_lr_scheduler else None,
        )

        # Eval after epoch (train split)
        train_set_dict = run_epoch(
            model,
            matcher,
            object_detector,
            dsg_generator,
            train_loader,
            train_dataset,
            None,
            device,
            None,
            epoch,
            train=False,
        )

        # Eval after epoch (test split)
        test_set_dict = run_epoch(
            model,
            matcher,
            object_detector,
            dsg_generator,
            test_loader,
            test_dataset,
            None,
            device,
            None,
            epoch,
            train=False,
        )

        end_time = time.time()
        epoch_duration = end_time - start_time

        # --------------------------------------------------------------
        # Console summary
        # --------------------------------------------------------------
        print()
        print("#" * 50)
        print(f"Evaluation after epoch {epoch}")

        if arg_method == "two_head_prediction":
            print(
                f"Train set: ["
                f"Loss Combined = {train_set_dict['loss_combined']:.4f}, "
                f"Loss Action Name = {train_set_dict['loss_action_name']:.4f}, "
                f"Loss Argument 1 = {train_set_dict['loss_arg1']:.4f}, "
                f"Loss Argument 2 = {train_set_dict['loss_arg2']:.4f}, "
                f"Accuracy Combined = {train_set_dict['acc_combined']:.4f}, "
                f"Accuracy Action Name = {train_set_dict['acc_action_name']:.4f}, "
                f"Accuracy Argument 1 = {train_set_dict['acc_arg1']:.4f}, "
                f"Accuracy Argument 2 = {train_set_dict['acc_arg2']:.4f}"
                f"]"
            )
            print()
            print(
                f"Test set: ["
                f"Loss Combined = {test_set_dict['loss_combined']:.4f}, "
                f"Loss Action Name = {test_set_dict['loss_action_name']:.4f}, "
                f"Loss Argument 1 = {test_set_dict['loss_arg1']:.4f}, "
                f"Loss Argument 2 = {test_set_dict['loss_arg2']:.4f}, "
                f"Accuracy Combined = {test_set_dict['acc_combined']:.4f}, "
                f"Accuracy Action Name = {test_set_dict['acc_action_name']:.4f}, "
                f"Accuracy Argument 1 = {test_set_dict['acc_arg1']:.4f}, "
                f"Accuracy Argument 2 = {test_set_dict['acc_arg2']:.4f}"
                f"]"
            )
        else:
            print(
                f"Train set: ["
                f"Loss Combined = {train_set_dict['loss_combined']:.4f}, "
                f"Loss Action Name = {train_set_dict['loss_action_name']:.4f}, "
                f"Loss Arguments = {train_set_dict['loss_args']:.4f}, "
                f"Accuracy Combined = {train_set_dict['acc_combined']:.4f}, "
                f"Accuracy Action Name = {train_set_dict['acc_action_name']:.4f}, "
                f"Accuracy Arguments = {train_set_dict['acc_args']:.4f}"
                f"]"
            )
            print()
            print(
                f"Test set: ["
                f"Loss Combined = {test_set_dict['loss_combined']:.4f}, "
                f"Loss Action Name = {test_set_dict['loss_action_name']:.4f}, "
                f"Loss Arguments = {test_set_dict['loss_args']:.4f}, "
                f"Accuracy Combined = {test_set_dict['acc_combined']:.4f}, "
                f"Accuracy Action Name = {test_set_dict['acc_action_name']:.4f}, "
                f"Accuracy Arguments = {test_set_dict['acc_args']:.4f}"
                f"]"
            )

        print(f"Time per Epoch = {epoch_duration:.2f} seconds")
        print("#" * 50)

        # --------------------------------------------------------------
        # W&B logging
        # --------------------------------------------------------------
        log_dict = {
            "epoch": epoch,
            "time/epoch_seconds": epoch_duration,

            # train metrics
            "train/loss_combined": train_set_dict["loss_combined"],
            "train/loss_action_name": train_set_dict["loss_action_name"],
            "train/acc_combined": train_set_dict["acc_combined"],
            "train/acc_action_name": train_set_dict["acc_action_name"],
            "train/acc_args": train_set_dict.get("acc_args", 0.0),
            "train/acc_arg1": train_set_dict.get("acc_arg1", 0.0),
            "train/acc_arg2": train_set_dict.get("acc_arg2", 0.0),
            "train/f1_action_macro": train_set_dict["f1_action_name_macro"],

            # test metrics
            "test/loss_combined": test_set_dict["loss_combined"],
            "test/loss_action_name": test_set_dict["loss_action_name"],
            "test/acc_combined": test_set_dict["acc_combined"],
            "test/acc_action_name": test_set_dict["acc_action_name"],
            "test/acc_args": test_set_dict.get("acc_args", 0.0),
            "test/acc_arg1": test_set_dict.get("acc_arg1", 0.0),
            "test/acc_arg2": test_set_dict.get("acc_arg2", 0.0),
            "test/f1_action_macro": test_set_dict["f1_action_name_macro"],
        }

        # Optional metrics depending on mode
        if "loss_args" in train_set_dict:
            log_dict["train/loss_args"] = train_set_dict["loss_args"]
        if "loss_arg1" in train_set_dict:
            log_dict["train/loss_arg1"] = train_set_dict["loss_arg1"]
        if "loss_arg2" in train_set_dict:
            log_dict["train/loss_arg2"] = train_set_dict["loss_arg2"]
        if "f1_args_macro" in train_set_dict:
            log_dict["train/f1_args_macro"] = train_set_dict["f1_args_macro"]

        if "loss_args" in test_set_dict:
            log_dict["test/loss_args"] = test_set_dict["loss_args"]
        if "loss_arg1" in test_set_dict:
            log_dict["test/loss_arg1"] = test_set_dict["loss_arg1"]
        if "loss_arg2" in test_set_dict:
            log_dict["test/loss_arg2"] = test_set_dict["loss_arg2"]
        if "f1_args_macro" in test_set_dict:
            log_dict["test/f1_args_macro"] = test_set_dict["f1_args_macro"]

        # Current LR
        if isinstance(optimizer, torch.optim.Optimizer):
            log_dict["lr"] = optimizer.param_groups[0]["lr"]

        wandb.log(log_dict)

        # --------------------------------------------------------------
        # Optional learning curve visualization
        # --------------------------------------------------------------
        visualize_learning_curve = True
        if visualize_learning_curve:
            try:
                from lib.tools.visualize_training_performance import plot_args_seperately, plot_args_combined

                if arg_method == "two_head_prediction":
                    plot_args_seperately()
                else:
                    plot_args_combined()
            except:
                pass

        # --------------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------------
        if use_early_stopping:
            early_stopper.step(test_set_dict["acc_combined"])
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

    wandb.finish()
    return (train_set_dict, test_set_dict)


if __name__ == "__main__":
    run_training()
