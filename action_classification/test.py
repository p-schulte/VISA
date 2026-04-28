from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

# Add project root (one level up) to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.config_loader import CONFIG
from lib.dataloader.dataloader import BlocksworldSequences
from lib.net.stgcn import STGCN

from action_classification.train import run_epoch, custom_cuda_collate_fn

from dsg_generator.lib.config import Config as STTranConfig
from dsg_generator.lib.matcher import HungarianMatcher
from dsg_generator.lib.object_detector import detector
from dsg_generator.lib.sttran import STTran


# ---------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------
def print_epoch_results(train_set_dict: dict, test_set_dict: dict) -> None:
    print(f"\n{'#' * 60}")

    base_keys = [
        "loss_combined",
        "loss_action_name",
        "loss_args",
        "acc_combined",
        "acc_action_name",
        "acc_args",
        "f1_action_name_macro",
        "f1_args_macro",
    ]

    # Append dynamic per-class keys in stable order
    keys = list(base_keys)
    for k in train_set_dict.keys():
        if ("args_acc_" in k or "type_acc_" in k) and k not in keys:
            keys.append(k)

    rows = []
    for key in keys:
        tr_val = train_set_dict.get(key, None)
        te_val = test_set_dict.get(key, None)

        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else ""

        rows.append(
            {
                "Metric": key,
                "Train": _fmt(tr_val),
                "Test": _fmt(te_val),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))
    print(f"{'#' * 60}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure repo root is importable (kept behavior; just cleaner)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # Config / parameters
    # -----------------------------------------------------------------
    conf = CONFIG.ac
    dsg_conf = CONFIG.dsg

    batch_size = conf["batch_size"]
    temporal_kernel_size = conf["temporal_kernel_size"]
    num_nodes = CONFIG.ac["num_nodes"][CONFIG.dsg["DATASET_NAME"]]
    num_features = conf["num_features"]

    arg_method = conf["stgcn_model"]["argument_prediction"]["method"]
    optimizer_name = conf["stgcn_model"]["argument_prediction"]["parameters"][arg_method]["optimizer"]
    print(f"Optimizer: {optimizer_name}")

    # Note: these optimizer params are parsed but not used in this eval script.
    if optimizer_name == "Adam":
        _ = conf["optimizer_params"]["Adam"]["learning_rate"]
        _ = conf["optimizer_params"]["Adam"]["weight_decay"]
    elif optimizer_name == "AdamW":
        _ = conf["optimizer_params"]["AdamW"]["learning_rate"]
        _ = conf["optimizer_params"]["AdamW"]["weight_decay"]
    elif optimizer_name == "SGD":
        _ = conf["optimizer_params"]["SGD"]["learning_rate"]
        _ = conf["optimizer_params"]["SGD"]["momentum"]
        _ = conf["optimizer_params"]["SGD"]["weight_decay"]
    elif optimizer_name == "RMSprop":
        _ = conf["optimizer_params"]["RMSprop"]["learning_rate"]
        _ = conf["optimizer_params"]["RMSprop"]["alpha"]
        _ = conf["optimizer_params"]["RMSprop"]["weight_decay"]

    # Paths
    dataset_dir = dsg_conf["data_path"]
    obj_det_model_path = dsg_conf["detector_model_path"]
    dsg_model_path = dsg_conf["dsg_model_path"]
    ac_model_path = dsg_conf["ac_model_path"]

    # -----------------------------------------------------------------
    # Datasets / loaders
    # -----------------------------------------------------------------
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

    spatial_kernel_size = len(train_dataset.spatial_relationships) + 1  # +1 for self-loop

    # -----------------------------------------------------------------
    # Action classification model (STGCN)
    # -----------------------------------------------------------------
    model = STGCN(
        num_classes=train_dataset.num_classes,
        in_channels=num_features,
        spatial_kernel_size=spatial_kernel_size,
        temporal_kernel_size=temporal_kernel_size,
        num_nodes=num_nodes,
    )
    model.eval()

    ckpt = torch.load(ac_model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    print()
    print("*" * 50)
    print(f"AC checkpoint {ac_model_path} is loaded")
    print("*" * 50)

    model.to(device)

    # -----------------------------------------------------------------
    # Matcher, detector, DSG generator
    # -----------------------------------------------------------------
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

    print(f"DSG-Generator checkpoint {dsg_model_path} is loaded")
    print("*" * 50)

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    print("\n" * 3)
    print("Evaluating model performance...")

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
        0,
        train=False,
        print_json=True,
        json_name="train_results.json",
    )

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
        0,
        train=False,
        print_json=True,
        json_name="test_results.json",
    )

    # -----------------------------------------------------------------
    # Console summary (kept output identical structure)
    # -----------------------------------------------------------------
    print()
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

    print(test_set_dict)
    print_epoch_results(train_set_dict, test_set_dict)
