from fasterRCNN.lib.model.utils.config import cfg
from config.config_loader import CONFIG

def visualize_ground_truth(
    im_data,
    gt_boxes_frame,            # (N,5) [x1,y1,x2,y2,label] for this frame
    filename="gt_visualization.png",
    frame_idx=0,
    class_names=None,          # optional: list of class names for labels
    return_image=False
):

    import cv2
    import numpy as np
    import torch
    # ---- image tensor -> numpy (BGR for OpenCV) ----
    im_np = im_data[frame_idx].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)

    # Be tolerant if cfg.PIXEL_MEANS is not in scope
    _pixel_means = None
    try:
        _ = cfg.PIXEL_MEANS  # will NameError if cfg not defined
        _pixel_means = cfg.PIXEL_MEANS
    except Exception:
        pass
    
    if _pixel_means is not None:
        # e.g., model preproc used RGB-mean subtraction
        im_np = im_np + _pixel_means
        im_np = np.clip(im_np, 0, 255)
    else:
        # assume input in [0,1] → scale to [0,255]
        # (change to *255.0 only if your tensors are definitely in [0,1])
        if im_np.max() <= 1.0:
            im_np = im_np * 255.0
        im_np = np.clip(im_np, 0, 255)

    canvas = im_np.astype(np.uint8)[..., ::-1].copy()  # RGB->BGR
    
    # ---- GT boxes to numpy & filter invalids (from padding) ----
    if torch.is_tensor(gt_boxes_frame):
        gt_np = gt_boxes_frame.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt_boxes_frame)

    if gt_np.ndim == 1:
        gt_np = gt_np.reshape(1, -1)
        
    # keep only boxes with positive area
    if gt_np.shape[1] >= 4:
        valid = (gt_np[:, 2] > gt_np[:, 0]) & (gt_np[:, 3] > gt_np[:, 1])
        gt_np = gt_np[valid]

    # ---- draw GT (blue) ----
    # ---- draw GT (blue) ----

    for g in gt_np:
        # make sure row is 1D and coerce each coord to a scalar
        row = np.asarray(g).squeeze()

        # skip malformed rows
        if row.ndim != 1 or row.shape[0] < 4:
            continue

        def as_int_scalar(val):
            a = np.asarray(val).squeeze()
            # take the first element if somehow >0D but still length-1
            if a.ndim > 0:
                if a.size == 0:
                    raise ValueError("empty value")
                a = a.reshape(-1)[0]
            return int(round(float(a)))

        try:
            x1 = as_int_scalar(row[0])
            y1 = as_int_scalar(row[1])
            x2 = as_int_scalar(row[2])
            y2 = as_int_scalar(row[3])
        except Exception:
            # bad coords → skip row
            continue

        # filter invalid/degenerate boxes
        if not (x2 > x1 and y2 > y1):
            continue

        # Correct OpenCV order: (top-left) → (bottom-right)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # optional label
        lab = -1
        if row.shape[0] > 4:
            try:
                lab_val = np.asarray(row[4]).squeeze()
                # guard np.isnan for non-float types
                is_nan = False
                try:
                    is_nan = bool(np.isnan(lab_val))
                except TypeError:
                    is_nan = False
                if not is_nan:
                    lab = int(round(float(lab_val))) if np.ndim(lab_val) == 0 else lab
            except Exception:
                pass

        if lab >= 0:
            text = str(class_names[lab]) if (class_names and 0 <= lab < len(class_names)) else f"GT:{lab}"
            yy = max(0, y1 - 6)
            cv2.putText(canvas, text, (x1, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # print(f"GT: {len(gt_np)} boxes")
    
    if return_image:
        return canvas
    else:
        cv2.imwrite(filename, canvas)
        return filename



def visualize_object_detection(im_data, entry, dataset, filename = "predictions_visualization.png", frame_idx=0, original_id=False, box_to_tracklet=None, return_image=False):
    import cv2
    import numpy as np

    """
    Visualizes predicted bounding boxes on the input image.
    
    :param im_data: Tensor of shape (batch_size, 3, H, W) containing the image.
    :param entry: Dictionary containing the detected 'boxes', 'labels', and 'scores'.
    :param class_names: List of class names corresponding to label indices.
    """
    USE_TRACKLET_IDS = box_to_tracklet is not None
    # Convert im_data (Tensor) to a NumPy image
    im_data_np = im_data[frame_idx].detach().cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    
    # Normalize image if needed (assumes input is in [0,1] range)
    im_data_np = im_data_np.astype(np.float32) + cfg.PIXEL_MEANS
    im_data_np = np.clip(im_data_np, 0, 255).astype(np.uint8)
    im_data_np = im_data_np[..., ::-1]

    # Copy image for visualization
    im2show = np.copy(im_data_np)

    # Extract predicted boxes, labels, and scores
    first_idx = (entry['boxes'][:, 0] == frame_idx).nonzero(as_tuple=True)[0][0].item()
    pred_boxes = entry['boxes'][entry['boxes'][:, 0] == frame_idx].cpu().numpy()  # Convert tensor to numpy
    pred_labels = entry['labels'][entry['boxes'][:, 0] == frame_idx].cpu().numpy()  # Class labels
    pred_scores = entry['scores'][entry['boxes'][:, 0] == frame_idx].cpu().numpy()  # Confidence scores

    # Loop through predictions and draw them on the image
    for i in range(len(pred_boxes)):
        x1, y1, x2, y2 = map(int, pred_boxes[i][1:])  # Extract bbox coordinates (ignore first value)
        label_idx = int(pred_labels[i])  # Class index
        score = pred_scores[i]  # Confidence score

        # Choose a color for the bounding box
        color = (0, 255, 0)  # Green for predictions
        cv2.rectangle(im2show, (x1, y2), (x2, y1), color, 2)

        # Label with class name and confidence
        label_text = f"{i + first_idx if original_id else i}"
        if USE_TRACKLET_IDS:
            label_text = f"{box_to_tracklet[i]}"

        cv2.putText(im2show, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # print(f"Object {i}: {label_text} with score {score:.3f}")

    
    # Either save or return
    if return_image:
        return im2show
    else:
        cv2.imwrite(filename, im2show)
        return filename



def visualize_predictions_console(pred, dataset, thresh=0.0, frame_idx=0):

        import torch
        print('*'*40)

        print("Predicted objects:")
        num_rows_until_first_img1 = (pred['boxes'][:, 0] == frame_idx).sum().item() # objects in the first frame
        for i, box in enumerate(pred['boxes'][:num_rows_until_first_img1]):
            label = dataset.object_classes[pred['labels'][i].item()]
            score = pred['scores'][i].item()
            print(f"    -> Object {i}: {label} with score {score:.3f}")

        print("Predicted relationships:")
        extraction = pred['pair_idx'][pred['im_idx'] == frame_idx]
        first_idx = (pred['im_idx'] == frame_idx).nonzero(as_tuple=True)[0][0].item()
        prev_box_num = 0
        if frame_idx > 0:
            prev_box_num = (pred['boxes'][:, 0] < frame_idx).sum().item()
        extraction = extraction - prev_box_num
        for i, pair in enumerate(extraction):
            subj = dataset.object_classes[pred['labels'][pair[0]].item()]
            obj = dataset.object_classes[pred['labels'][pair[1]].item()]

            spatial_scores = torch.sigmoid(pred['spatial_distribution'])[first_idx + i]
            for j, spatial_score in enumerate(spatial_scores):
                if spatial_score < thresh:
                    continue
                spatial_pred = dataset.spatial_relationships[j]
                print(f"Relationship {i}: {subj}(id {pair[0]}) -> {obj}(id {pair[1]}) with spatial relationship '{spatial_pred}' (score {spatial_score:.3f})")   
        print('*'*40)


import networkx as nx
import subprocess
import torch

def visualize_scene_graph(pred, dataset, filename, dot_filename='scene_graph.dot', thresh=0.0, frame_idx=0, box_to_tracklet=None, return_image=False):
    """
    Create a scene graph from pred and dataset using networkx,
    export it as a DOT file, and generate an image using Graphviz.
    
    Args:
        pred (dict): Prediction dictionary containing keys such as 'boxes', 'labels',
                     'scores', 'pair_idx', 'im_idx', and 'spatial_distribution'.
        dataset: Dataset object with attributes `object_classes` and `spatial_relationships`.
        dot_filename (str): Filename for the DOT file.
        filename (str): Filename for the output image (e.g., PNG).
        thresh (float): Threshold for including spatial relationship edges.
    """
    USE_TRACKLET_IDS = box_to_tracklet is not None

    # Create directed multigraph
    G = nx.MultiDiGraph()
    
    # Determine number of objects from the first frame (im_idx==0)
    num_objs = (pred['boxes'][:, 0] == frame_idx).sum().item()
    
    # Add nodes: each node's label comes from dataset.object_classes
    labels = pred['labels']
    action_objects = []
    action_object_names = []
    for i in range(num_objs):
        score = pred['scores'][i].item()
        label = f'"{i}: {dataset.object_classes[labels[i].item()]}"'
        if USE_TRACKLET_IDS:
            label = f'"{box_to_tracklet[i]}: {dataset.object_classes[labels[i].item()]}"'

            G.add_node(box_to_tracklet[i], label=label, score=score)
            # print("Object {}: {} with score {:.3f}".format(box_to_tracklet[i], dataset.object_classes[labels[i].item()], score))
            action_objects.append(box_to_tracklet[i])
            action_object_names.append(dataset.object_classes[labels[i].item()])
        else:
            G.add_node(i, label=label, score=score)
            # print("Object {}: {} with score {:.3f}".format(i, dataset.object_classes[labels[i].item()], score))
            action_objects.append(i)
            action_object_names.append(dataset.object_classes[labels[i].item()])
    
    # Add edges: for relationships in the first frame (im_idx==0)
    relations = {}
    extraction = pred['pair_idx'][pred['im_idx'] == frame_idx]
    first_idx = (pred['im_idx'] == frame_idx).nonzero(as_tuple=True)[0][0].item()
    prev_box_num = 0
    if frame_idx > 0:
        prev_box_num = (pred['boxes'][:, 0] < frame_idx).sum().item()
    extraction = extraction - prev_box_num
    for i, pair in enumerate(extraction):
        spatial_scores = torch.sigmoid(pred['spatial_distribution'])[first_idx + i]
        for j, spatial_score in enumerate(spatial_scores):
            if spatial_score < thresh:
                continue
            spatial_pred = dataset.spatial_relationships[j]
            # Add an edge with label and score attributes
            if USE_TRACKLET_IDS:
                G.add_edge(box_to_tracklet[pair[0].item()], box_to_tracklet[pair[1].item()],
                           label=spatial_pred, score=spatial_score.item())
                # print(f"Relationship {i}: {box_to_tracklet[pair[0].item()]} -> {box_to_tracklet[pair[1].item()]} with spatial relationship '{spatial_pred}' (score {spatial_score.item():.3f})")
                try:
                    relations[spatial_pred].append((box_to_tracklet[pair[0].item()], box_to_tracklet[pair[1].item()]))
                except KeyError:
                    relations[spatial_pred] = [(box_to_tracklet[pair[0].item()], box_to_tracklet[pair[1].item()])]
            else:
                G.add_edge(pair[0].item(), pair[1].item(),
                           label=spatial_pred, score=spatial_score.item())
    
    nx.drawing.nx_pydot.write_dot(G, dot_filename)

    if return_image:
        # render to memory instead of file
        import pydot
        import cv2
        import numpy as np
        graphs = pydot.graph_from_dot_file(dot_filename)
        png_bytes = graphs[0].create_png()
        nparr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return action_objects, action_object_names, relations, img
    else:
        subprocess.run(["dot", "-Tpng", dot_filename, "-o", filename], check=True)
        return action_objects, action_object_names, relations

























import cv2
import numpy as np

def _resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

def _put_title(img, text):
    if not text:
        return img
    pad = 28  # space for text
    canvas = np.full((img.shape[0] + pad, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[pad:, :] = img
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return canvas

def hstack_three(pred_img, gt_img, sg_img, gap=10, bg_color=(255,255,255),
                 titles=("Pred", "GT", "SceneGraph")):
    """Resize to a common height and stack horizontally with optional titles."""
    # choose a target height (e.g., the min to avoid upscaling too much)
    target_h = min(pred_img.shape[0], gt_img.shape[0], sg_img.shape[0])

    pred_resized = _resize_to_height(pred_img, target_h)
    gt_resized   = _resize_to_height(gt_img,   target_h)
    sg_resized   = _resize_to_height(sg_img,   target_h)

    # add titles (optional)
    pred_t = _put_title(pred_resized, titles[0] if titles else None)
    gt_t   = _put_title(gt_resized,   titles[1] if titles else None)
    sg_t   = _put_title(sg_resized,   titles[2] if titles else None)

    # make a gap strip
    def gap_strip(h, w):
        return np.full((h, w, 3), bg_color, dtype=np.uint8)

    H = max(pred_t.shape[0], gt_t.shape[0], sg_t.shape[0])
    # pad to the same height (in case title rows differ slightly)
    def pad_h(img):
        if img.shape[0] == H: return img
        pad = H - img.shape[0]
        return np.vstack([img, gap_strip(pad, img.shape[1])])

    pred_t = pad_h(pred_t)
    gt_t   = pad_h(gt_t)
    sg_t   = pad_h(sg_t)

    g = gap_strip(H, gap)
    combo = np.hstack([pred_t, g, gt_t, g, sg_t])
    return combo

import cv2
import numpy as np

def grid_three(pred_img, gt_img, sg_img, gap=10, bg_color=(255,255,255),
               titles=("Pred", "GT", "SceneGraph")):
    """
    Layout:
      [ Pred | GT ]
      [   SceneGraph (spans full width)   ]
    """
    def gap_h(h, w): return np.full((h, w, 3), bg_color, dtype=np.uint8)
    def gap_v(h, w): return np.full((h, w, 3), bg_color, dtype=np.uint8)

    # --- Top row: normalize heights for pred & gt ---
    target_h = min(pred_img.shape[0], gt_img.shape[0])   # avoid upscaling
    pred_r = _resize_to_height(pred_img, target_h)
    gt_r   = _resize_to_height(gt_img,   target_h)

    # Add titles (optional)
    pred_t = _put_title(pred_r, titles[0] if titles else None)
    gt_t   = _put_title(gt_r,   titles[1] if titles else None)

    # Pad to same height (due to title rows)
    Htop = max(pred_t.shape[0], gt_t.shape[0])
    def pad_h(img):
        if img.shape[0] == Htop: return img
        pad = Htop - img.shape[0]
        return np.vstack([img, gap_h(pad, img.shape[1])])

    pred_t = pad_h(pred_t)
    gt_t   = pad_h(gt_t)

    # Compose top row with a vertical gap between
    top = np.hstack([pred_t, gap_v(Htop, gap), gt_t])

    # --- Bottom row: resize scene-graph to full top width ---
    top_w = top.shape[1]
    # keep aspect ratio by width
    scale = top_w / float(sg_img.shape[1])
    new_h = max(1, int(round(sg_img.shape[0] * scale)))
    sg_resized = cv2.resize(sg_img, (top_w, new_h), interpolation=cv2.INTER_AREA)
    sg_titled = _put_title(sg_resized, titles[2] if titles else None)

    # --- Stack top + horizontal gap + bottom ---
    combo = np.vstack([top, gap_h(gap, top_w), sg_titled])
    return combo









def compose_summary_grid(
    num: int,
    pred1: np.ndarray, gt1: np.ndarray,
    pred2: np.ndarray, gt2: np.ndarray,
    sg1: np.ndarray, sg2: np.ndarray,
    pred_action_name: str,
    gt_action_name: str,
    gap: int = 10,
    bg_color=(255, 255, 255),
    title_font_scale: float = 0.7,
    body_font_scale: float = 0.7,
    font_thickness: int = 2
) -> np.ndarray:
    import cv2, numpy as np
    from typing import Optional, Tuple

    # ---------- helpers ----------
    def gap_h(h, w): 
        if h <= 0 or w <= 0: 
            # avoid negative or zero dims (return minimal band)
            h = max(1, h); w = max(1, w)
        return np.full((h, w, 3), bg_color, dtype=np.uint8)
    def gap_v(h, w): 
        return gap_h(h, w)

    def _ensure_color(img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            # Defensive: make a tiny placeholder to avoid crashes
            return np.full((1, 1, 3), 127, np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 1:
            return np.repeat(img, 3, axis=2)
        return img

    def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        target_h = max(1, int(target_h))
        h, w = img.shape[:2]
        if h == target_h:
            return img
        scale = target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        interp = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
        return cv2.resize(img, (new_w, target_h), interpolation=interp)

    def _draw_text_center(canvas: np.ndarray, text: str, baseline_pt: Tuple[int, int],
                          font_scale: float, bold: bool = False):
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = font_thickness + (1 if bold else 0)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = max(5, (canvas.shape[1] - tw) // 2)
        y = baseline_pt[1]
        cv2.putText(canvas, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def _draw_text_left(canvas: np.ndarray, text: str, x: int, y: int,
                        font_scale: float, thickness: int):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def _put_title(img: np.ndarray, title: Optional[str]) -> np.ndarray:
        if not title:
            return img
        pad = max(20, int(28 * title_font_scale))
        band = gap_h(pad, img.shape[1])
        out = np.vstack([band, img])
        _draw_text_center(out, title, (0, pad - 5), font_scale=title_font_scale, bold=True)
        return out

    def _wrap_text(text: str, max_width_px: int, font_scale: float, thickness: int) -> list:
        font = cv2.FONT_HERSHEY_SIMPLEX
        words = text.split()
        lines, cur = [], ""
        for w in words:
            test = w if not cur else (cur + " " + w)
            (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw <= max_width_px:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur: lines.append(cur)
        return lines or [""]

    # ---------- sanitize inputs ----------
    pred1 = _ensure_color(pred1); pred2 = _ensure_color(pred2)
    gt1   = _ensure_color(gt1);   gt2   = _ensure_color(gt2)
    sg1   = _ensure_color(sg1);   sg2   = _ensure_color(sg2)

    # ---------- make rows ----------
    # Header (temp width; we’ll repad later)
    header_text = f"{num} -> {num + 1}"
    header_h = max(30, int(40 * title_font_scale))
    # take a reasonable provisional width
    provisional_w = max(img.shape[1] for img in [pred1, pred2, gt1, gt2, sg1, sg2])
    header_band = gap_h(header_h, provisional_w)
    _draw_text_center(header_band, header_text, (0, header_h - 10), font_scale=title_font_scale, bold=True)

    # Row 1: pred1 | pred2
    h1 = min(pred1.shape[0], pred2.shape[0])
    pred1_t = _put_title(_resize_to_height(pred1, h1), f"Pred {num}")
    pred2_t = _put_title(_resize_to_height(pred2, h1), f"Pred {num+1}")
    H1 = max(pred1_t.shape[0], pred2_t.shape[0])
    if pred1_t.shape[0] < H1: pred1_t = np.vstack([pred1_t, gap_h(H1 - pred1_t.shape[0], pred1_t.shape[1])])
    if pred2_t.shape[0] < H1: pred2_t = np.vstack([pred2_t, gap_h(H1 - pred2_t.shape[0], pred2_t.shape[1])])
    row1 = np.hstack([pred1_t, gap_v(H1, gap), pred2_t])

    # Row 2: gt1 | gt2
    h2 = min(gt1.shape[0], gt2.shape[0])
    gt1_t = _put_title(_resize_to_height(gt1, h2), f"GT {num}")
    gt2_t = _put_title(_resize_to_height(gt2, h2), f"GT {num+1}")
    H2 = max(gt1_t.shape[0], gt2_t.shape[0])
    if gt1_t.shape[0] < H2: gt1_t = np.vstack([gt1_t, gap_h(H2 - gt1_t.shape[0], gt1_t.shape[1])])
    if gt2_t.shape[0] < H2: gt2_t = np.vstack([gt2_t, gap_h(H2 - gt2_t.shape[0], gt2_t.shape[1])])
    row2 = np.hstack([gt1_t, gap_v(H2, gap), gt2_t])

    # Compute a single global max width (never shrink anything)
    max_width = max(header_band.shape[1], row1.shape[1], row2.shape[1])

    # Resize SGs to that width
    def _resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        if w == target_w:
            return img
        scale = target_w / float(w)
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(img, (target_w, new_h), interpolation=interp)

    annot = "Predicted" if not CONFIG.ac['use_gt_dsg'] else "Ground Truth"
    sg1_titled = _put_title(_resize_to_width(sg1, max_width), f"{annot} SceneGraph {num}")
    sg2_titled = _put_title(_resize_to_width(sg2, max_width), f"{annot} SceneGraph {num+1}")

    # Pad rows up to max_width (only grow; never try to shrink)
    def pad_right_to(img: np.ndarray, W: int) -> np.ndarray:
        if img.shape[1] >= W:
            return img
        pad_w = W - img.shape[1]
        return np.hstack([img, gap_v(img.shape[0], pad_w)])

    header_band = pad_right_to(header_band, max_width)
    row1 = pad_right_to(row1, max_width)
    row2 = pad_right_to(row2, max_width)

    # Footer (Pred vs. GT)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, th_ref), _ = cv2.getTextSize("Ag", font, body_font_scale, font_thickness)
    th = max(12, th_ref)

    pred_text = f"Action - Pred. :  {pred_action_name}".strip()
    gt_text   = f"Action - GT    :  {gt_action_name}".strip()

    lines_pred = _wrap_text(pred_text, max_width - 20, body_font_scale, font_thickness)
    lines_gt   = _wrap_text(gt_text,   max_width - 20, body_font_scale, font_thickness)

    all_lines = lines_pred + lines_gt
    footer_h = 16 + len(all_lines) * (th + 8) + 8
    footer_band = gap_h(footer_h, max_width)

    y = 16 + th
    for line in all_lines:
        _draw_text_left(footer_band, line, 10, y, body_font_scale, font_thickness)
        y += th + 8


    # Assemble
    combo = np.vstack([
        header_band,
        gap_h(gap, max_width),
        row1,
        gap_h(gap, max_width),
        row2,
        gap_h(gap, max_width),
        sg1_titled,
        gap_h(gap, max_width),
        sg2_titled,
        gap_h(gap, max_width),
        footer_band
    ])
    return combo
