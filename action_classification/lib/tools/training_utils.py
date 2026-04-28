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

import wandb

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

# Add project root (one level up) to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.config_loader import CONFIG

from lib.net.stgcn import STGCN
from lib.dataloader.dataloader import BlocksworldSequences
from lib.tools.dsg_construction_utils import (
    construct_dynamic_scene_graph_with_tracking,
    construct_dynamic_scene_graph_with_tracking_one_partition,
    gt_to_tracklet_indices,
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


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def custom_cuda_collate_fn(batch: List[Any]) -> List[Any]:
    """
    Collate function that leaves sequences intact.

    Useful when each item is already a sequence / variable-length structure and
    you want to handle batching manually in the training loop.
    """
    return batch


# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------
class EarlyStopping:
    """
    Early stopping utility to prevent overfitting and shorten training.

    Monitors a score (higher is better, e.g., accuracy / mAP / F1). If the score
    doesn't improve by at least `min_delta` for `patience` consecutive steps,
    training can be stopped.

    Args:
        patience: Number of evaluations without improvement before stopping.
        min_delta: Minimum improvement required to reset patience.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False

    def step(self, current_score: float) -> None:
        """
        Update early stopping state given the latest score.

        Args:
            current_score: The most recent metric value. Higher is better.
        """
        if self.best_score is None:
            self.best_score = current_score
            return

        improved = current_score > (self.best_score + self.min_delta)
        if improved:
            self.best_score = current_score
            self.counter = 0
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True







def visualize_predictions(im_info, im_data, entry_dsg_generator, dsg_gt_annotation,
                          action_gt_annotation, dataset, b, ARGUMENT_LOSS,
                          pred_action_name, pred_arguments,action_names, 
                          matcher, sttran_conf,
                          action_names_gt_ids, arguments_gt_tracklet_ids, max_arity,
                          gt_boxes):
    """
    Visualizes object detection and predicted vs. ground truth action semantics for a batch element.

    This function renders visualizations for each frame in a video sequence, including:
    - Detected object bounding boxes with associated tracklet IDs.
    - Dynamic Scene Graph (DSG) visualizations.
    - Console printout comparing predicted and ground truth actions and their arguments.

    Parameters:
    -----------
    im_info : Tensor
        Image metadata, including dimensions and scaling.
    im_data : Tensor
        Input image tensor used for inference.
    entry_dsg_generator : dict
        Output from the DSG generator containing detected objects, tracklets, features, and boxes.
    dsg_gt_annotation : list
        List of ground truth DSG annotations per frame.
    action_gt_annotation : list
        List of ground truth action annotations per frame (used for evaluating predictions).
    dataset : Dataset
        The dataset instance providing class and label information.
    b : int
        Current batch index (used for naming output files).
    ARGUMENT_LOSS : str
        Type of argument loss used ("CrossEntropyLoss" or "BCEWithLogitsLoss").
    pred_action_name : Tensor
        Predicted action name scores (logits) for each frame.
    pred_arguments : list of Tensors
        Predicted arguments; interpretation depends on `ARGUMENT_LOSS`.
    action_names : list
        List of all possible action name strings.
    matcher : HungarianMatcher
        Matcher used for aligning predicted and ground truth object indices.
    sttran_conf : STTranConfig
        Configuration used for DSG generation.
    action_names_gt_ids : Tensor
        Ground truth action name indices for each frame.
    arguments_gt_tracklet_ids : Tensor
        Ground truth argument tracklet IDs for each frame and arity.
    max_arity : int
        Maximum number of arguments per action (e.g., 2 for binary actions).

    Returns:
    --------
    None
        Saves visualizations to disk and prints formatted action prediction vs. ground truth.
    """
    import dsg_generator.lib.visualize_predictions as viz_pred

    # Refining the detected IDs to tracklet IDs
    REFINE_IDS = not CONFIG.ac['use_gt_dsg']
    REFINE_IDS = True
    if REFINE_IDS:
        get_sequence(entry_dsg_generator, dsg_gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, sttran_conf.mode, GENERAL_COARSE_TRACKING=True)
        
        box_to_tracklet = list()
        indices = entry_dsg_generator['indices']
        for i in range(len(entry_dsg_generator['boxes'])):
            index = -1
            for i_t, tracklet in enumerate(indices):
                if i in tracklet:
                    index = i_t
                    break
            box_to_tracklet.append(index)

    
    # Print the predicted vs. ground truth actions in the console
    pred_action_name_index = torch.max(torch.sigmoid(pred_action_name), dim=1).indices
    if ARGUMENT_LOSS == 'CrossEntropyLoss':
        pred_arguments_indices = [torch.max(torch.sigmoid(args), dim=1).indices for args in pred_arguments]
    elif ARGUMENT_LOSS == 'BCEWithLogitsLoss':
        pred_arguments_indices = torch.sigmoid(pred_arguments[0]).topk(max_arity, dim=1).indices
        pred_arguments_indices_ar1 = torch.sigmoid(pred_arguments[0]).topk(1, dim=1).indices


    def get_frame_id(frame_idx):
        import copy
        start = (entry_dsg_generator['boxes'][:, 0] == frame_idx).nonzero(as_tuple=True)[0][0].item()
        end = (entry_dsg_generator['boxes'][:, 0] == frame_idx).nonzero(as_tuple=True)[0][-1].item()
        btt_excerpt = copy.deepcopy(box_to_tracklet[start:end+1]) if REFINE_IDS else None
        start_2 = (entry_dsg_generator['boxes'][:, 0] == frame_idx+1).nonzero(as_tuple=True)[0][0].item()
        end_2 = (entry_dsg_generator['boxes'][:, 0] == frame_idx+1).nonzero(as_tuple=True)[0][-1].item()
        btt_excerpt_2 = copy.deepcopy(box_to_tracklet[start_2:end_2+1]) if REFINE_IDS else None

        # Render predictions of objects in the image
        pred1 = viz_pred.visualize_object_detection(
        im_data, entry_dsg_generator, dataset,
            frame_idx=frame_idx,
            filename=f"visualization/det_{b}_{frame_idx+1}.png",
            box_to_tracklet=btt_excerpt,
            return_image=True
        )
        pred2 = viz_pred.visualize_object_detection(
        im_data, entry_dsg_generator, dataset,
            frame_idx=frame_idx+1,
            filename=f"visualization/det_{b}_{frame_idx+1}.png",
            box_to_tracklet=btt_excerpt_2,
            return_image=True
        )

        # Render Ground Truth objects in the image
        gt1 = viz_pred.visualize_ground_truth(
            im_data, gt_boxes[frame_idx],
            filename=f"visualization/gt_{b}_{frame_idx+1}.png",
            frame_idx=frame_idx,
            class_names=dataset.object_classes,
            return_image=True
        )
        gt2 = viz_pred.visualize_ground_truth(
            im_data, gt_boxes[frame_idx+1],
            filename=f"visualization/gt_{b}_{frame_idx+1}.png",
            frame_idx=frame_idx+1,
            class_names=dataset.object_classes,
            return_image=True
        )

        # Render Scene Graphs
        sg1 = viz_pred.visualize_scene_graph(
            entry_dsg_generator, dataset,
            f"visualization/dsg_{b}_{frame_idx+1}.png",
            frame_idx=frame_idx, thresh=0.1,
            box_to_tracklet=btt_excerpt,
            return_image=True
        )[3]
        sg2 = viz_pred.visualize_scene_graph(
            entry_dsg_generator, dataset,
            f"visualization/dsg_{b}_{frame_idx+1}.png",
            frame_idx=frame_idx+1, thresh=0.1,
            box_to_tracklet=btt_excerpt_2,
            return_image=True
        )[3]


        # Getting the action + args
        gt_action_name_str = action_names[action_names_gt_ids[frame_idx]] # GT Action
        args = [arguments_gt_tracklet_ids[arity][frame_idx] for arity in range(max_arity)]
        if args == [0, 0]:
            args = []
        gt_args_str = ""
        for arg in args:
            gt_args_str += f"{arg}, "
        if gt_args_str != "":
            gt_args_str = gt_args_str[:-2]
        gt_action_str = f"{gt_action_name_str}({gt_args_str})"
        if gt_action_name_str == "":
            gt_action_str = "No action"

        curr_action_name = pred_action_name_index[frame_idx].item() # Pred Action
        if ARGUMENT_LOSS == 'CrossEntropyLoss':
            curr_args = [str(args[frame_idx].item()) for args in pred_arguments_indices]
            pred_arguments_str = ", ".join(curr_args)        
        elif ARGUMENT_LOSS == 'BCEWithLogitsLoss':
            if max_arity == 2:
                indices_to_choose = pred_arguments_indices_ar1 if (args[0] == args[1]).item() else pred_arguments_indices
            else:
                indices_to_choose = pred_arguments_indices
            pred_arguments_str = "SET: " + ", ".join([str(ele.item()+1) for ele in indices_to_choose[frame_idx]])

        pred_action_str = f"{action_names[curr_action_name]}({pred_arguments_str})"
        if curr_action_name == 0:
            pred_action_str = "No action"



        

        # Finally rendering the visualization grid
        import cv2
        from dsg_generator.lib.visualize_predictions import compose_summary_grid
        combo = compose_summary_grid(frame_idx+1,
                            pred1=pred1, gt1=gt1,
                            pred2=pred2, gt2=gt2,
                            sg1=sg1, sg2=sg2,
                            pred_action_name=pred_action_str,
                            gt_action_name=gt_action_str
                           )



        # Save
        output_dir = f"/u/paul.schulte/wlink/logs/ac_results/{CONFIG.dsg['DATASET_NAME']}/{CONFIG.ac['run_name']}/example_visualization/{dataset.mode}/"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/merged_grid_b{b}_{frame_idx+1}->{frame_idx+2}.png", combo)

    # Visualize object detection results and DSG generation results
    for frame_idx in range(len(dsg_gt_annotation)-1):
        try:
            get_frame_id(frame_idx)
        except Exception as e:
            print(f"Error during visualization of frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()








def obtain_json_annotation(im_info, im_data, entry_dsg_generator, dsg_gt_annotation,
                          action_gt_annotation, dataset, b, ARGUMENT_LOSS,
                          pred_action_name, pred_arguments,action_names, 
                          matcher, sttran_conf,
                          action_names_gt_ids, arguments_gt_tracklet_ids, max_arity):
                          
    import dsg_generator.lib.visualize_predictions as viz_pred

    # Refining the detected IDs to tracklet IDs
    get_sequence(entry_dsg_generator, dsg_gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, sttran_conf.mode, GENERAL_COARSE_TRACKING=True)
    
    REFINE_IDS = True
    box_to_tracklet = list()
    indices = entry_dsg_generator['indices']
    for i in range(len(entry_dsg_generator['boxes'])):
        index = -1
        for i_t, tracklet in enumerate(indices):
            if i in tracklet:
                index = i_t
                break
        box_to_tracklet.append(index)


    def get_frame_id(frame_idx):
        import copy
        start = (entry_dsg_generator['boxes'][:, 0] == frame_idx).nonzero(as_tuple=True)[0][0].item()
        end = (entry_dsg_generator['boxes'][:, 0] == frame_idx).nonzero(as_tuple=True)[0][-1].item()
        btt_excerpt = copy.deepcopy(box_to_tracklet[start:end+1]) if REFINE_IDS else None   
        
        # viz_pred.visualize_object_detection(im_data, entry_dsg_generator, dataset, frame_idx=frame_idx, filename=f"visualization/det_{b}_{frame_idx+1}.png", box_to_tracklet = btt_excerpt)
        return viz_pred.visualize_scene_graph(entry_dsg_generator, dataset, f"visualization/dsg_{b}_{frame_idx+1}.png", frame_idx=frame_idx, thresh=0.1, box_to_tracklet = btt_excerpt)        

    # Visualize object detection results and DSG generation results
    action_objects_list, action_object_names_list, relations_list = [], [], []
    frame_plus_index_to_global_index = []
    frame_ranges = []
    for frame_idx in range(len(dsg_gt_annotation)):
        import copy
        start = (entry_dsg_generator['boxes'][:, 0] == frame_idx).nonzero(as_tuple=True)[0][0].item()
        end = (entry_dsg_generator['boxes'][:, 0] == frame_idx).nonzero(as_tuple=True)[0][-1].item()
        frame_ranges.append((start, end))
        btt_excerpt = copy.deepcopy(box_to_tracklet[start:end+1]) if REFINE_IDS else None
        local_list = [i-1 for i in btt_excerpt]
        frame_plus_index_to_global_index.append(local_list)
        
        # viz_pred.visualize_object_detection(im_data, entry_dsg_generator, dataset, frame_idx=frame_idx, filename=f"visualization/det_{b}_{frame_idx+1}.png", box_to_tracklet = btt_excerpt)
        action_objects, action_object_names, relations = viz_pred.visualize_scene_graph(entry_dsg_generator, dataset, f"visualization/dsg_{b}_{frame_idx+1}.png", frame_idx=frame_idx, thresh=0.1, box_to_tracklet = btt_excerpt)
        action_objects_list.append(action_objects)
        action_object_names_list.append(action_object_names)
        relations_list.append(relations)
        


    # Print the predicted vs. ground truth actions in the console
    pred_action_name_index = torch.max(torch.sigmoid(pred_action_name), dim=1).indices
    if ARGUMENT_LOSS == 'CrossEntropyLoss':
        pred_arguments_indices = [torch.max(torch.sigmoid(args), dim=1).indices for args in pred_arguments]
    elif ARGUMENT_LOSS == 'BCEWithLogitsLoss':
        pred_arguments_indices = torch.sigmoid(pred_arguments[0]).topk(max_arity, dim=1).indices


    thresh = 0.5
    mask = (torch.sigmoid(pred_arguments[0]) > thresh)
    per_row_true_indices = [
        mask[i].nonzero(as_tuple=True)[0].tolist()
        for i in range(mask.size(0))
    ]
    pred_arguments_indices = per_row_true_indices

    state_action_representations = []

    for frame_idx, _ in enumerate(action_gt_annotation[:-1]):
        # Ground truth action
        gt_action_name_str = action_names[action_names_gt_ids[frame_idx]]
        args = [arguments_gt_tracklet_ids[arity][frame_idx] for arity in range(max_arity)]
        if args == [0, 0]:
            args = []
        gt_args_str = ""
        for arg in args:
            gt_args_str += f"{arg}, "
        if gt_args_str != "":
            gt_args_str = gt_args_str[:-2]
        gt_action_str = f"{gt_action_name_str}({gt_args_str})"
        if gt_action_name_str == "":
            gt_action_str = "No action"

        # Predicted action
        curr_action_name = pred_action_name_index[frame_idx].item()

        if ARGUMENT_LOSS == 'CrossEntropyLoss':
            curr_args = [str(args[frame_idx].item()) for args in pred_arguments_indices]
            pred_arguments_str = ", ".join(curr_args)        
        elif ARGUMENT_LOSS == 'BCEWithLogitsLoss':
            # define ordering of arguments based on box changes
            order = []
            for i in range(min(len(frame_plus_index_to_global_index[frame_idx]), len(frame_plus_index_to_global_index[frame_idx+1]))):
                try:
                    index_in_curr_frame = frame_plus_index_to_global_index[frame_idx].index(i)
                    index_in_next_frame = frame_plus_index_to_global_index[frame_idx+1].index(i)


                    box_curr = entry_dsg_generator['boxes'][entry_dsg_generator['boxes'][:, 0] == frame_idx][index_in_curr_frame][1:]
                    box_next = entry_dsg_generator['boxes'][entry_dsg_generator['boxes'][:, 0] == frame_idx+1][index_in_next_frame][1:]

                    cx1 = (box_curr[0] + box_curr[2]) / 2
                    cy1 = (box_curr[1] + box_curr[3]) / 2
                    cx2 = (box_next[0] + box_next[2]) / 2
                    cy2 = (box_next[1] + box_next[3]) / 2

                    distance = torch.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                    order.append( (distance.item(), i) )
                except:
                    pass
            order_raw = sorted(order, key=lambda x: x[0])
            order_raw.reverse()
            order = [ele[1] for ele in order_raw]


            # obtain correct order
            vals = pred_arguments_indices[frame_idx]
            for ele in vals:
                if ele not in order:
                    order.append(ele)
            sorted_vals = sorted(vals, key=lambda x: order.index(x))


            # pred_arguments_str = [str(ele.item()+1) for ele in pred_arguments_indices[frame_idx]]
            pred_arguments_str = [str(ele+1) for ele in sorted_vals]

        
        # convert ids to strings
        relations = relations_list[frame_idx]
        for rel in relations:
            updated = []
            for ele in relations[rel]:
                lst = []
                for id in ele:
                    lst.append(str(id))
                updated.append(lst)
            relations[rel] = updated


        state_representation = {}
        state_representation.update(relations)
        state_representation["detected_objects"] = [ str(ele) for ele in action_objects_list[frame_idx] ]
        state_representation["detected_object_names"] = action_object_names_list[frame_idx]

        # Collect it in a list
        state_action_representations.append({
            "state" : state_representation,
            "action-name" : action_names[curr_action_name],
            "action-objects" : pred_arguments_str if action_names[curr_action_name] != "none" else []
        })


    return state_action_representations



def visualize_adj_with_labels(A,
                            labels,
                            class_names=None,
                            title=None,
                            save_path=None,
                            show=True):
    """
    Visualize a single adjacency matrix A as a graph.

    Parameters
    ----------
    A : (N, N) torch.Tensor or np.ndarray
        Adjacency matrix for one relation at one timestep.
        Non-zero entries => edges.
    labels : (N,) torch.Tensor or np.ndarray
        Class ID for each node. Use -1 for padded / invalid nodes.
    class_names : list[str] or None
        Optional mapping from class id -> human-readable name for legend.
    title : str or None
        Title for the plot.
    save_path : str or None
        If given, figure is saved to this path.
    show : bool
        If True, plt.show() is called. Otherwise the figure is closed.
    """

    # ----- convert to numpy on CPU -----
    if torch.is_tensor(A):
        A = A.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    A = np.asarray(A)
    labels = np.asarray(labels)

    assert A.ndim == 2, f"A must be 2-D [N,N], got {A.shape}"
    N = A.shape[0]
    assert labels.shape[0] == N, "labels must have length N"

    # ----- pick only valid nodes (label >= 0) -----
    valid_nodes = np.where(labels >= 0)[0]
    if len(valid_nodes) == 0:
        print("No valid nodes (all labels < 0); nothing to plot.")
        return

    # ----- build graph -----
    G = nx.Graph()

    # add nodes with their class as attribute
    for i in valid_nodes:
        G.add_node(int(i), cls=int(labels[i]))

    # add edges where adjacency is non-zero
    for i in valid_nodes:
        for j in valid_nodes:
            if i == j:
                continue
            if A[i, j] != 0:
                G.add_edge(int(i), int(j))

    print(f"#nodes: {G.number_of_nodes()}, #edges: {G.number_of_edges()}")

    # ----- color mapping by class -----
    unique_classes = sorted({G.nodes[n]["cls"] for n in G.nodes()})
    cmap = colormaps.get_cmap("tab10")  # modern Matplotlib API

    # assign a distinct color to each class id
    denom = max(1, len(unique_classes) - 1)
    class_to_color = {
        c: cmap(k / denom) for k, c in enumerate(unique_classes)
    }

    node_colors = [class_to_color[G.nodes[n]["cls"]] for n in G.nodes()]

    # ----- layout & drawing -----
    pos = nx.spring_layout(G, seed=0)  # or nx.circular_layout(G)

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=500,
        edgecolors="k",
        linewidths=0.5,
    )
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)

    # label nodes with "index(class)"
    node_labels = {n: f"{n}({G.nodes[n]['cls']})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # ----- legend -----
    legend_handles = []
    for c in unique_classes:
        if class_names is not None and 0 <= c < len(class_names):
            name = class_names[c]
        else:
            name = f"class {c}"
        legend_handles.append(
            Patch(facecolor=class_to_color[c], edgecolor="k", label=name)
        )
    plt.legend(
        handles=legend_handles,
        title="Classes",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path) if save_path is not None else None



def _pca_and_scatter_indexed(X, y_pairs, title, class_names, out_dir, stem, max_points=20000):
    """
    PCA scatter for (class_id, node_idx) pairs.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Features to visualize.
    y_pairs : np.ndarray, shape (N, 2)
        Each row = [class_id, node_index].
        class_id == -1 will be ignored.
    class_names : list or dict
        Maps class ids to class names.
    out_dir : str
        Output directory for saving the figure.
    stem : str
        File name stem.
    """
    import numpy as np, os, datetime
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    y_pairs = np.asarray(y_pairs)
    assert y_pairs.ndim == 2 and y_pairs.shape[1] == 2, \
        f"Expected (N,2), got {y_pairs.shape}"

    # --- Mask invalid class ids (e.g., -1 paddings) ---
    mask = y_pairs[:, 0] >= 0
    X = X[mask]
    y_pairs = y_pairs[mask]
    if X.shape[0] == 0:
        print(f"[PCA] No valid points for {stem}; skipped.")
        return

    # --- Subsample for readability ---
    if X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], max_points, replace=False)
        X = X[idx]
        y_pairs = y_pairs[idx]

    # --- Map each unique (class_id, node_idx) pair to a compact label ---
    unique_pairs, inverse = np.unique(y_pairs, axis=0, return_inverse=True)
    y_compact = inverse  # shape (N,)

    # --- Build human-readable labels like "Block/7" ---
    legend_labels = []
    for cls_id, node_idx in unique_pairs:
        if isinstance(class_names, dict):
            cls_name = class_names.get(int(cls_id), str(int(cls_id)))
        elif isinstance(class_names, (list, tuple)) and 0 <= int(cls_id) < len(class_names):
            cls_name = class_names[int(cls_id)]
        else:
            cls_name = str(int(cls_id))
        legend_labels.append(f"{cls_name}/{int(node_idx)}")

    # --- PCA to 2D ---
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y_compact, cmap='tab20', s=6, alpha=0.75)

    uniq = np.unique(y_compact)
    handles = []
    legend_names = []
    for u in uniq:
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='',
                                markersize=6, markerfacecolor=scatter.cmap(scatter.norm(u))))
        legend_names.append(legend_labels[int(u)] if int(u) < len(legend_labels) else str(int(u)))

    plt.legend(handles, legend_names, title="ObjClass/Index",
            bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[PCA] Saved {path}")

    # optional TensorBoard TSV export if you already have this helper
    _export_projector_tsv(X2, y_compact, out_dir, stem)






def _export_projector_tsv(X2d, labels, out_dir, stem):
    tens_path = os.path.join(out_dir, f"{stem}_tensor.tsv")
    meta_path = os.path.join(out_dir, f"{stem}_metadata.tsv")
    np.savetxt(tens_path, X2d, delimiter="\t")
    with open(meta_path, "w") as f:
        for v in labels:
            f.write(f"{int(v)}\n")
    return tens_path, meta_path

def _pca_and_scatter(X, y, title, label_names, out_dir, stem, max_points=20000):
    mask = y >= 0
    X = X[mask]
    y = y[mask]

    if X.shape[0] == 0:
        print(f"[PCA] No valid points for {stem}; skipped.")
        return

    if X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], max_points, replace=False)
        X = X[idx]
        y = y[idx]

    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="tab20", s=6, alpha=0.75)

    uniq = np.unique(y)
    handles = []
    legend_labels = []
    for cls in uniq:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=scatter.cmap(scatter.norm(cls)),
            )
        )
        if label_names is not None and 0 <= cls < len(label_names):
            legend_labels.append(label_names[int(cls)])
        else:
            legend_labels.append(str(int(cls)))

    plt.legend(
        handles,
        legend_labels,
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{stem}.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"[PCA] Saved {png_path}")

    _export_projector_tsv(X2, y, out_dir, stem)