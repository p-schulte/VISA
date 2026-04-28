import torch
import numpy as np

def _to_np_bbox(b):
    """Accept bbox as numpy array, list/tuple, or torch tensor; return np.ndarray shape (4,)."""
    if isinstance(b, np.ndarray):
        out = b
    elif isinstance(b, torch.Tensor):
        out = b.detach().cpu().numpy()
    else:
        out = np.asarray(b)
    out = out.astype(float)  # keep float like in your target tensor
    assert out.shape[-1] == 4, f"bbox must have 4 coords, got shape {out.shape}"
    return out

def frames_to_bbox_tensor(frames, device="cpu", dtype=torch.float32, require_visible=False):
    """
    frames: [[{'frame': '...'}, obj1, obj2, ...], ...]
      where each obj has a 'bbox' field like array([x1,y1,x2,y2]) and optional 'visible'.
    Returns: torch.Tensor [N,5] with rows [frame_idx, x1, y1, x2, y2]
    """
    rows = []
    for k, pack in enumerate(frames):
        # pack[0] is the header {'frame': '...'}
        for obj in pack[1:]:
            if 'bbox' not in obj:
                continue
            if require_visible and obj.get('visible', 'true') != 'true':
                continue
            bb = _to_np_bbox(obj['bbox'])
            rows.append([float(k), bb[0], bb[1], bb[2], bb[3]])
    if not rows:
        return torch.empty((0,5), dtype=dtype, device=device)
    t = torch.tensor(rows, dtype=dtype)
    return t.to(device)


def extract_classes(frames, device="cpu"):
    """
    frames: list of frames in 'this structure'
      each frame = [{'frame': '...'}, obj1, obj2, ...]
      each obj has a 'class' key (int)

    Returns: torch.Tensor [N] with class IDs in order.
    """
    classes = []
    for pack in frames:
        for obj in pack[1:]:  # skip the {'frame': ...} dict
            classes.append(int(obj['class']))
    return torch.tensor(classes, device=device)


# matching of two things

import torch
import numpy as np

def _to_tensor5(x, device=None, dtype=torch.float32):
    """
    Coerce x into a [N,5] torch.Tensor.
    Accepts:
      - torch.Tensor [N,5]  (any device/dtype)
      - np.ndarray  [N,5]
      - list/tuple of rows, or list of tensors (will cat/stack)
    """
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        if len(x) == 0:
            t = torch.empty((0, 5), dtype=dtype)
        elif isinstance(x[0], torch.Tensor):
            # list of tensors; cat along first dim after ensuring 2D
            parts = [xi.reshape(-1, xi.shape[-1]) for xi in x]
            t = torch.cat(parts, dim=0)
        else:
            # list of lists/tuples/numbers
            t = torch.tensor(x, dtype=dtype)
    else:
        raise TypeError(f"Unsupported type for boxes: {type(x)}")
    if t.numel() == 0:
        t = t.reshape(0, 5).to(dtype=dtype)
    assert t.shape[-1] == 5, f"expected [...,5], got {tuple(t.shape)}"
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if device is not None and t.device != torch.device(device):
        t = t.to(device)
    return t

def _split_by_frame(t: torch.Tensor):
    """
    t: [N,5] with [frame, x1,y1,x2,y2]
    returns: dict frame_idx -> {'rows': [global_row_indices], 'boxes': np.ndarray [M,4]}
    """
    t_cpu = t.detach().cpu()
    frames = {}
    for i in range(t_cpu.shape[0]):
        f = int(t_cpu[i, 0].item())
        if f not in frames:
            frames[f] = {'rows': [], 'boxes': []}
        frames[f]['rows'].append(i)
        frames[f]['boxes'].append(t_cpu[i, 1:5].numpy())
    for f in frames:
        frames[f]['boxes'] = np.stack(frames[f]['boxes'], axis=0) if frames[f]['boxes'] else np.zeros((0,4))
    return frames

def _cost_matrix(A: np.ndarray, B: np.ndarray, metric="l1"):
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A_exp = A[:, None, :]   # [m,1,4]
    B_exp = B[None, :, :]   # [1,n,4]
    if metric == "l2":
        C = np.sqrt(((A_exp - B_exp) ** 2).sum(axis=-1))
    else:
        C = np.abs(A_exp - B_exp).sum(axis=-1)
    return C.astype(np.float32)

def match_bboxes_per_frame(t1, t2, metric: str = "l1", device=None, dtype=torch.float32):
    """
    Robust matcher:
      - t1, t2 can be Tensor / ndarray / list (even list of tensors per batch)
      - returns per-frame Hungarian matching (falls back to greedy if scipy missing)

    Output dict:
      frame_idx -> {
         'pairs': [(i1, i2, cost), ...],      # global row indices into coerced t1/t2
         'unmatched_t1': [i1, ...],
         'unmatched_t2': [i2, ...]
      }
    """
    t1 = _to_tensor5(t1, device=device, dtype=dtype)
    t2 = _to_tensor5(t2, device=device, dtype=dtype)

    F1 = _split_by_frame(t1)
    F2 = _split_by_frame(t2)

    out = {}
    all_frames = sorted(set(F1.keys()) | set(F2.keys()))

    try:
        from scipy.optimize import linear_sum_assignment
        has_scipy = True
    except Exception:
        has_scipy = False

    for f in all_frames:
        rows1 = F1.get(f, {'rows': [], 'boxes': np.zeros((0,4))})['rows']
        A = F1.get(f, {'rows': [], 'boxes': np.zeros((0,4))})['boxes']
        rows2 = F2.get(f, {'rows': [], 'boxes': np.zeros((0,4))})['rows']
        B = F2.get(f, {'rows': [], 'boxes': np.zeros((0,4))})['boxes']

        m, n = A.shape[0], B.shape[0]
        pairs, unmatched1, unmatched2 = [], [], []

        if m == 0 and n == 0:
            out[f] = {'pairs': [], 'unmatched_t1': [], 'unmatched_t2': []}
            continue

        C = _cost_matrix(A, B, metric=metric)

        if m == 0:
            unmatched2 = rows2.copy()
        elif n == 0:
            unmatched1 = rows1.copy()
        else:
            if has_scipy:
                r_idx, c_idx = linear_sum_assignment(C)  # optimal
            else:
                # Greedy fallback
                r_idx, c_idx = [], []
                used_r, used_c = set(), set()
                for _ in range(min(m, n)):
                    best = None
                    best_val = float('inf')
                    for i in range(m):
                        if i in used_r: continue
                        for j in range(n):
                            if j in used_c: continue
                            v = C[i, j]
                            if v < best_val:
                                best_val = v; best = (i, j)
                    if best is None: break
                    i, j = best
                    r_idx.append(i); c_idx.append(j)
                    used_r.add(i); used_c.add(j)

            used_r_set = set()
            used_c_set = set()
            for i, j in zip(r_idx, c_idx):
                pairs.append((rows1[i], rows2[j], float(C[i, j])))
                used_r_set.add(i); used_c_set.add(j)
            unmatched1.extend(rows1[i] for i in range(m) if i not in used_r_set)
            unmatched2.extend(rows2[j] for j in range(n) if j not in used_c_set)

        out[f] = {'pairs': pairs, 'unmatched_t1': unmatched1, 'unmatched_t2': unmatched2}

    return out



def reorder_features_to_t1(features_t2, mapping, len_t1, fill_value=0.0):
    """
    Reorder features (aligned with t2 / predictions) into t1 (GT) order.

    Args:
      features_t2: [N2, D] tensor (prediction features, order = t2)
      mapping: dict from match_bboxes_per_frame
      len_t1: number of GT boxes
      fill_value: value for unmatched GT boxes (use float('nan') if you prefer)

    Returns:
      features_t1: [len_t1, D], aligned to GT order
      idx: [len_t1] long, indices into t2 (or -1 if unmatched)
      mask: [len_t1] bool, True if matched
    """
    device = features_t2.device
    D = features_t2.shape[1]
    idx = torch.full((len_t1,), -1, dtype=torch.long, device=device)

    # mapping['pairs'] contains (i1, i2, cost) with i1=t1 index, i2=t2 index
    for f in mapping.values():
        for i1, i2, _ in f['pairs']:
            idx[i1] = i2

    mask = idx >= 0
    features_t1 = torch.full((len_t1, D), fill_value, dtype=features_t2.dtype, device=device)
    features_t1[mask] = features_t2[idx[mask]]
    return features_t1, idx, mask








# extracting features:
import torch
import torch.nn.functional as F
from dsg_generator.fasterRCNN.lib.model.roi_layers import ROIAlign
from dsg_generator.lib.draw_rectangles.draw_rectangles import draw_union_boxes  # your util

@torch.no_grad()
def populate_image_derived_fields(entry, *, stride=16, roi_out=(7,7), force_feat_dim=2048):
    """
    Mutates `entry` in-place to add:
      - entry['features']   : (N, D) per-object features (D=force_feat_dim)
      - entry['union_box']  : (P, 5) [img,x1,y1,x2,y2] in pixels
      - entry['union_feat'] : (P, C, 7, 7) per-pair union ROI features
      - entry['spatial_masks'] : (P, 2, 27, 27) subj/obj masks, mean-centered

    Requires in `entry`:
      boxes (N,5) normalized as [frame_id, x1,y1,x2,y2]
      im_info ([W,H,W,H])
      pair_idx (P,2), im_idx (P,)
      fmaps (B,C,Hf,Wf)
    """
    assert all(k in entry for k in ['boxes','im_info','fmaps','pair_idx','im_idx'])
    device = entry['boxes'].device
    fmaps  = entry['fmaps']
    C      = fmaps.shape[1]

    # ROIAlign configured like in your STTran
    roi_align = ROIAlign(roi_out, 1.0/float(stride), 0).to(device)

    # ---- Per-object features ----
    boxes_px = entry['boxes'].clone()
    boxes_px[:, 1:] = boxes_px[:, 1:] * entry['im_info']          # to pixels
    obj_rois   = boxes_px                                         # (N,5) [img,x1,y1,x2,y2] in px
    obj_roi_ft = roi_align(fmaps, obj_rois)                       # (N, C, 7, 7)
    obj_pooled = obj_roi_ft.mean(dim=(2,3))                       # (N, C)

    # Make sure downstream sees the expected dim (often 2048 in your code)
    if C == force_feat_dim:
        entry['features'] = obj_pooled
    elif C < force_feat_dim:
        entry['features'] = F.pad(obj_pooled, (0, force_feat_dim - C))  # zero-pad
    else:
        # quick projection to 2048 without defining a Module
        W = torch.zeros(C, force_feat_dim, device=device)
        W[:force_feat_dim, :force_feat_dim] = torch.eye(force_feat_dim, device=device)
        entry['features'] = obj_pooled @ W

    # ---- Per-pair union box + union features ----
    if entry['pair_idx'].numel() > 0:
        b1 = boxes_px[entry['pair_idx'][:, 0], 1:5]
        b2 = boxes_px[entry['pair_idx'][:, 1], 1:5]
        xy1 = torch.min(b1[:, :2], b2[:, :2])
        xy2 = torch.max(b1[:, 2:],  b2[:, 2:])
        union_box = torch.cat([entry['im_idx'].unsqueeze(1), xy1, xy2], dim=1)  # (P,5) in px
        entry['union_box']  = union_box
        entry['union_feat'] = roi_align(fmaps, union_box)                        # (P, C, 7, 7)
    else:
        entry['union_box']  = torch.zeros(0, 5, device=device)
        entry['union_feat'] = torch.zeros(0, C, roi_out[0], roi_out[1], device=device)

    # ---- Spatial masks (27×27, subj/obj channels) ----
    if entry['pair_idx'].numel() > 0:
        # draw_union_boxes expects concatenated subj|obj boxes in *normalized* coords
        pair_rois = torch.cat(
            [entry['boxes'][entry['pair_idx'][:,0], 1:5],
             entry['boxes'][entry['pair_idx'][:,1], 1:5]], dim=1
        ).cpu().numpy()
        entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5, device=device)
    else:
        entry['spatial_masks'] = torch.zeros(0, 2, 27, 27, device=device)

    return entry







# getting pairs:
import torch

@torch.no_grad()
def pairs_from_dsg_gt(dsg_gt_annotation, device, include_self_pairs=True):
    """
    Build pair_idx and im_idx directly from dsg_gt_annotation.

    Assumes the same object order per frame as frames_to_bbox_tensor:
      - for each frame: dsg_gt_annotation[t] is a list whose first item is {'frame': <name>}
        and the remaining items are object dicts (in-order).
      - global indexing concatenates frames in order.

    Returns:
      pair_idx: (P, 2) LongTensor of global object indices
      im_idx  : (P,)   FloatTensor with the frame id (0..T-1) per pair
      obj_indices_per_frame: list of LongTensor with the global indices for each frame
    """
    # 1) Collect counts and global index ranges per frame
    counts = []
    for frame in dsg_gt_annotation:
        # objects start at index 1 (index 0 is the {'frame': ...} header)
        counts.append(max(len(frame) - 1, 0))
    offsets = [0]
    for c in counts[:-1]:
        offsets.append(offsets[-1] + c)
    offsets = torch.tensor(offsets, dtype=torch.long)

    # 2) Build per-frame global index tensors
    obj_indices_per_frame = []
    for t, c in enumerate(counts):
        if c == 0:
            obj_indices_per_frame.append(torch.zeros(0, dtype=torch.long))
        else:
            start = offsets[t].item()
            obj_indices_per_frame.append(torch.arange(start, start + c, dtype=torch.long))

    # 3) Build ordered pairs within each frame (+/- self pairs)
    pair_chunks = []
    im_chunks = []
    for t, idxs in enumerate(obj_indices_per_frame):
        n = idxs.numel()
        if n == 0:
            continue
        A, B = torch.meshgrid(idxs, idxs, indexing="ij")  # (n,n)
        if not include_self_pairs:
            mask = (A != B)
            A = A[mask]; B = B[mask]
        pairs_t = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)  # (p_t, 2)
        pair_chunks.append(pairs_t)
        im_chunks.append(torch.full((pairs_t.size(0),), float(t)))

    if len(pair_chunks) == 0:
        pair_idx = torch.zeros(0, 2, dtype=torch.long, device=device)
        im_idx   = torch.zeros(0, dtype=torch.float, device=device)
    else:
        pair_idx = torch.cat(pair_chunks, dim=0).to(device=device, dtype=torch.long)
        im_idx   = torch.cat(im_chunks,   dim=0).to(device=device, dtype=torch.float)

    # Move per-frame index lists to device too (handy for debugging/visualization)
    obj_indices_per_frame = [x.to(device) for x in obj_indices_per_frame]
    return pair_idx, im_idx, obj_indices_per_frame


import torch

import math

def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

# Examples:
pos_logit = logit(0.999)   # ≈ +6.9078
neg_logit = logit(0.001)   # ≈ -6.9078

@torch.no_grad()
def spatial_distribution_from_dsg_gt(
    dsg_gt_annotation,
    pair_idx: torch.Tensor,          # (P,2) global indices
    im_idx: torch.Tensor,            # (P,)  frame ids
    rel_classes=('at-robby','at','free','carry','different'),
    *,
    device=None,
    pos_logit: float=pos_logit,
    neg_logit: float=neg_logit,
    # treat these unary relations as *self-pair only* (i==j)
    unary_as_self=('at-robby','free'),
    # optional type gating: subj_classes, obj_classes per relation (None = no gating)
    type_filters=None,               # e.g. {'at':({1},{2}), 'carry':({1},{3}), 'different':({2},{2})}
):
    if device is None: device = pair_idx.device

    # counts & offsets to map global -> (frame, local)
    counts = [max(len(frame)-1, 0) for frame in dsg_gt_annotation]
    offsets = [0]
    for c in counts[:-1]: offsets.append(offsets[-1] + c)

    # cache per-frame info
    per_frame = []
    for frame in dsg_gt_annotation:
        objs   = frame[1:]
        unary  = [set(obj.get('unary_relationships', [])) for obj in objs]
        binary = [{k: set(v) for k,v in obj.get('binary_relationships',{}).items()} for obj in objs]
        classes= [int(obj.get('class', -1)) for obj in objs]          # 1=ball, 2=room, 3=gripper (your scheme)
        ids    = [obj.get('identifier', f'obj{j}') for j,obj in enumerate(objs)]
        per_frame.append((objs, unary, binary, classes, ids))

    P = pair_idx.size(0)
    R = len(rel_classes)
    out = torch.full((P, R), neg_logit, device=device)

    unary_as_self = set(unary_as_self)

    for p in range(P):
        t = int(im_idx[p].item())
        if counts[t] == 0: continue

        i_g, j_g = pair_idx[p].tolist()
        i_l = i_g - offsets[t]; j_l = j_g - offsets[t]
        if not (0 <= i_l < counts[t] and 0 <= j_l < counts[t]):  # safety
            continue

        _, unary, binary, classes, ids = per_frame[t]
        subj_id, obj_id = ids[i_l], ids[j_l]
        ci, cj = classes[i_l], classes[j_l]
        same = (i_l == j_l)

        for r_idx, rel in enumerate(rel_classes):
            # optional class gating
            if type_filters and rel in type_filters:
                subj_ok, obj_ok = type_filters[rel]
                if (subj_ok is not None and ci not in subj_ok) or (obj_ok is not None and cj not in obj_ok):
                    continue

            if rel == 'at-robby':
                # unary on room -> light only self-pair if set in unary_as_self
                if cj == 2 and ('at-robby' in unary[j_l]) and ((rel not in unary_as_self) or same):
                    out[p, r_idx] = pos_logit

            elif rel == 'at':
                if 'at' in binary[i_l] and (obj_id in binary[i_l]['at']):
                    out[p, r_idx] = pos_logit

            elif rel == 'free':
                # unary on gripper
                if ci == 3 and ('free' in unary[i_l]) and (('free' not in unary_as_self) or same):
                    out[p, r_idx] = pos_logit

            elif rel == 'carry':
                if 'carry' in binary[i_l] and (obj_id in binary[i_l]['carry']):
                    out[p, r_idx] = pos_logit

            elif rel == 'different':
                if ci == 2 and cj == 2:
                    if (obj_id in binary[i_l].get('different', set())) or (subj_id in binary[j_l].get('different', set())):
                        out[p, r_idx] = pos_logit

            else:
                # unknown relation: leave negative
                pass

    return out



import torch

@torch.no_grad()
def build_trivial_indices_from_boxes(boxes: torch.Tensor):
    """
    boxes: (N_total, 5) with [frame_id, x1, y1, x2, y2] (frame_id may be float)
    Returns:
      indices: list[LongTensor], where
         indices[0] is empty,
         indices[1+i] are tracklets linking the i-th object across all frames.
    Assumes:
      - All frames have the same number of objects.
      - Per-frame object order is consistent across frames.
    """
    device = boxes.device
    frame_ids = boxes[:, 0].long()
    uniq = frame_ids.unique(sorted=True)
    # collect per-frame global indices preserving order
    per_frame_idxs = [(frame_ids == t).nonzero(as_tuple=False).view(-1) for t in uniq]
    counts = torch.tensor([len(ix) for ix in per_frame_idxs], device=device)
    if not torch.all(counts == counts[0]):
        raise ValueError(f"Frames have differing counts: {counts.tolist()} — "
                         "cannot build trivial one-to-one indices safely.")

    T = len(per_frame_idxs)           # number of frames
    N = counts[0].item()              # objects per frame
    # global concatenation is assumed: frame0 then frame1 then ...
    # Build tracklets: for each slot i in 0..N-1, stack that slot across frames
    indices = [torch.empty(0, dtype=torch.long, device=device)]  # indices[0] empty
    base = per_frame_idxs[0][0].item()  # usually 0
    # If frames are strictly concatenated, per_frame_idxs[t] == torch.arange(t*N, (t+1)*N)
    for i in range(N):
        idxs = torch.stack([per_frame_idxs[t][i] for t in range(T)], dim=0)
        indices.append(idxs.to(device=device, dtype=torch.long))
    return indices


import math
import torch
from typing import Optional, Dict, Set, Tuple

def _logit_from_prob(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))

@torch.no_grad()
def spatial_distribution_from_dsg_generic(
    dsg_gt_annotation,
    pair_idx: torch.Tensor,                # (P,2) global object indices
    im_idx: torch.Tensor,                  # (P,)  frame ids (float/int)
    rel_classes,                           # list of relation names, e.g. dataset.relation_classes
    *,
    device: Optional[torch.device] = None,
    pos_prob: float = 0.999,               # -> ~ +6.9 logit
    neg_prob: float = 0.001,               # -> ~ -6.9 logit
    unary_on_self: bool = True,            # unary relations only on (i==j)
    class_gates: Optional[Dict[str, Tuple[Optional[Set[int]], Optional[Set[int]]]]] = None,
    symmetric_rels: Optional[Set[str]] = None,
    id_key: str = 'identifier',
):
    """
    Domain-agnostic construction of spatial_distribution logits (P, R).

    Positives:
      - Unary rel r: positive if (i==j) and r in unary[i] (if unary_on_self).
      - Binary rel r: positive if j.id in binary[i][r].
        If r in symmetric_rels, also accept i.id in binary[j][r].

    Everything else is negative.
    """
    if device is None:
        device = pair_idx.device

    pos_logit = _logit_from_prob(pos_prob)
    neg_logit = _logit_from_prob(neg_prob)

    # counts & offsets to map global -> (frame t, local)
    counts = [max(len(frame) - 1, 0) for frame in dsg_gt_annotation]
    offsets = [0]
    for c in counts[:-1]:
        offsets.append(offsets[-1] + c)

    # cache per-frame info
    per_frame = []
    for frame in dsg_gt_annotation:
        objs = frame[1:]
        unary = [set(obj.get('unary_relationships', [])) for obj in objs]
        # normalize binary to {rel: set(ids)}
        binary = []
        for obj in objs:
            bmap = {}
            for rel, ids in obj.get('binary_relationships', {}).items():
                bmap[rel] = set(ids)
            binary.append(bmap)
        classes = [int(obj.get('class', -1)) for obj in objs]
        ids     = [obj.get(id_key, 'obj{}'.format(j)) for j, obj in enumerate(objs)]
        per_frame.append((unary, binary, classes, ids))

    P = pair_idx.size(0)
    R = len(rel_classes)
    out = torch.full((P, R), neg_logit, device=device)

    if symmetric_rels is None:
        symmetric_rels = set()

    for p in range(P):
        t = int(im_idx[p].item())
        if counts[t] == 0:
            continue

        i_g, j_g = pair_idx[p].tolist()
        i_l = i_g - offsets[t]
        j_l = j_g - offsets[t]
        if not (0 <= i_l < counts[t] and 0 <= j_l < counts[t]):
            continue

        unary, binary, classes, ids = per_frame[t]
        ci, cj = classes[i_l], classes[j_l]
        id_i, id_j = ids[i_l], ids[j_l]
        same = (i_l == j_l)

        for r_idx, rel in enumerate(rel_classes):
            # optional class gating
            if class_gates is not None and rel in class_gates:
                subj_ok, obj_ok = class_gates[rel]
                if (subj_ok is not None and ci not in subj_ok) or (obj_ok is not None and cj not in obj_ok):
                    continue  # leave at neg_logit

            # unary
            if unary_on_self and same and (rel in unary[i_l] or rel in unary[j_l]):
                out[p, r_idx] = pos_logit
                continue

            # binary i -> j
            pos = False
            if rel in binary[i_l] and (id_j in binary[i_l][rel]):
                pos = True
            elif rel in symmetric_rels:
                if rel in binary[j_l] and (id_i in binary[j_l][rel]):
                    pos = True

            if pos:
                out[p, r_idx] = pos_logit

    return out







import torch
import torch.nn.functional as F
from lib.fpn.box_utils import center_size
from lib.word_vectors import obj_edge_vectors

@torch.no_grad()
def augment_features_to_2376(entry, object_classes, device=None, keep_dropout=False):
    """
    entry (mutated in-place):
      requires: entry['features'] (N,2048), entry['boxes'] (N,5 normalized), and either
                entry['distribution'] (N, C-1) or entry['labels'] (N,) with classes >=1.
      object_classes: list like dataset.object_classes with background at index 0.

    Produces:
      entry['features'] -> (N, 2376) = [2048 ROI | 200 semantic | 128 geometric]

    Notes:
      - Uses the same GloVe initializer as in ObjectClassifier for the 200-d semantics.
      - Geometry: Linear(4→128)+ReLU (+Dropout 0.1 if keep_dropout=True).
      - If 'distribution' missing, falls back to labels (one-hot over classes 1..C-1).
    """
    if device is None:
        device = entry['features'].device

    feats = entry['features']          # (N, 2048) now
    assert feats.dim() == 2 and feats.size(1) == 2048, f"expected (N,2048), got {feats.shape}"
    boxes = entry['boxes']             # (N,5) [frame, x1,y1,x2,y2] normalized

    # ---- 200-d semantic embedding using GloVe table over classes 1..C-1 ----
    C = len(object_classes)            # includes background at 0
    glove = obj_edge_vectors(object_classes[1:], wv_type='glove.6B', wv_dir='data', wv_dim=200).to(device)
    # distribution preferred; else build from labels
    if 'distribution' in entry and entry['distribution'] is not None:
        dist = entry['distribution'].to(device)            # (N, C-1)
        assert dist.size(1) == C-1, f"distribution second dim must be C-1 ({C-1})"
        sem200 = dist @ glove                              # (N,200)
    else:
        assert 'labels' in entry, "need 'labels' if no 'distribution' is provided"
        labels = entry['labels'].to(device).clamp(min=0)   # (N,)
        # one-hot over 1..C-1 -> index shift by -1, background(0) -> all-zeros
        dist = torch.zeros(labels.size(0), C-1, device=device)
        valid = labels > 0
        dist[valid, labels[valid] - 1] = 1.0
        sem200 = dist @ glove                               # (N,200)

    # ---- 128-d geometric embedding from normalized boxes (center-size) ----
    # Your ObjectClassifier does: BN(4) -> Linear(4,128) -> ReLU -> Dropout(0.1)
    # We mirror the Linear+ReLU (+ optional Dropout). Fresh weights are fine for debugging/GT paths.
    pos4 = center_size(boxes[:, 1:]).to(device)            # (N,4) in [0,1]
    # quick linear projection 4->128 (deterministic init)
    torch.manual_seed(0)
    W = torch.empty(4, 128, device=device); torch.nn.init.xavier_uniform_(W)
    b = torch.zeros(128, device=device)
    geo128 = F.linear(pos4, W.t(), b)
    geo128 = F.relu(geo128)
    if keep_dropout:
        geo128 = F.dropout(geo128, p=0.1, training=True)   # mimic Dropout(0.1)

    # ---- concat to 2376 ----
    entry['features'] = torch.cat([feats, sem200, geo128], dim=1)  # (N, 2376)



import torch
from lib.tools.gt_utils import extract_classes  # you already use this

@torch.no_grad()
def build_gt_distribution_from_dsg(
    dsg_gt_annotation,
    object_classes,              # e.g. dataset.object_classes, background at index 0
    device,
    eps_other: float = 1e-5      # tiny prob for non-GT classes (matches your 9e-06..1e-05)
):
    """
    Returns:
      distribution: (N, C-1) float32 probs over classes 1..C-1, aligned with frames_to_bbox_tensor order.
    """
    # Labels in the exact same global order as frames_to_bbox_tensor:
    labels = extract_classes(dsg_gt_annotation, device=device)  # (N,), values in {1..C-1}

    C_minus_1 = len(object_classes) - 1
    N = labels.numel()
    dist = torch.full((N, C_minus_1), eps_other, device=device, dtype=torch.float32)

    # Put almost all mass on the GT class:
    tgt = labels.clamp(min=1) - 1                         # shift to [0..C-2]
    off_count = max(C_minus_1 - 1, 0)
    p_tgt = 1.0 - eps_other * off_count                   # e.g. 1 - 2*1e-5 = 0.99998 (for 3 classes)
    dist[torch.arange(N, device=device), tgt] = p_tgt

    return dist
