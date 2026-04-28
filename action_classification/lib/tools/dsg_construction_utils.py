import torch



def construct_dynamic_scene_graph_with_tracking(entry_dsg, thresh=0.5, padding=0, num_rels=6):
    """Construct dynamic scene graph from the DSG-Generator output."""
    Xs = []
    Xs_labels = []
    As = []

    # Obtain tracklets
    tracklets = entry_dsg['indices']
    tracklets = [t for t in tracklets if t.numel() > 0]

    # Fill the lacking tracklets with -1 to ensure all tracklets have the same length
    max_len = max([len(t) for t in tracklets])
    for i, t in enumerate(tracklets):
        if len(t) < max_len:
            padd = torch.full((max_len - len(t),), -1, dtype=t.dtype, device=t.device)
            tracklets[i] = torch.cat((t, padd))

    num_tracklets = len(tracklets)
    len_tracklet = min([len(t) for t in tracklets])
    for t in tracklets:
        if len(t) != len_tracklet:
            #print("Tracklet length mismatch")
            pass #import pdb; pdb.set_trace()

    # Determine whether directed or undirected relationships
    from config.config_loader import CONFIG
    dataset_name = CONFIG.dsg['DATASET_NAME']
    undirected = False #CONFIG.ac['bidirectional_adjacency_matrix'][dataset_name] # used to be defind per dataset, but made univeral now

    # Number of frames
    counter= 0
    for frame_idx in range(len_tracklet):

        # node features and adjacency matrix of this frame
        X = []
        X_labels = []
        adj_mat_size = max(num_tracklets, padding)
        A = torch.zeros(num_rels, adj_mat_size, adj_mat_size)

        # Extract objects of current frame from tracklets in correct order
        det_id_in_frame = []
        for tr in tracklets:
            det_id_in_frame.append(tr[frame_idx])
        
        
        # Add current node features
        for i in det_id_in_frame:
            score = entry_dsg['scores'][i].item()
            label = entry_dsg['labels'][i].item()
            features = entry_dsg['features'][i]
            X.append(features)
            X_labels.append(label)

        for i in range(max(0, padding - num_tracklets)): # Padd features to always have the same dimension
            X.append(torch.zeros_like(entry_dsg['features'][0]))
            X_labels.append(-1) # Padding label
        X = torch.stack(X)

        
        # Construct current adjacency matrix
        method = 'fast' # 'fast' or 'old'
        if method == 'old':
            for i1, ele1 in enumerate(det_id_in_frame):
                for i2, ele2 in enumerate(det_id_in_frame):
                    # Apply a sigmoid to spatial_distribution scores
                    if [ele1, ele2] in entry_dsg['pair_idx'].tolist():
                        spat_dist_score_index = entry_dsg['pair_idx'].tolist().index([ele1, ele2])
                        spatial_scores = torch.sigmoid(entry_dsg['spatial_distribution'])[spat_dist_score_index]
                        for j, spatial_score in enumerate(spatial_scores):
                            if spatial_score < thresh:
                                continue
                            # Add an edge with label and score attributes
                            A[j, i1, i2] = 1
                            A[j, i2, i1] = 1

        elif method == 'fast':
            # Precompute everything outside the loops
            pair_idx_list = entry_dsg['pair_idx'].tolist()
            pair_idx_map = {tuple(pair): idx for idx, pair in enumerate(pair_idx_list)}
            spatial_distr_sigmoid = torch.sigmoid(entry_dsg['spatial_distribution'])

            for i1, ele1 in enumerate(det_id_in_frame):
                for i2, ele2 in enumerate(det_id_in_frame):
                    if (ele1.item(), ele2.item()) in pair_idx_map:
                        spat_dist_score_index = pair_idx_map[(ele1.item(), ele2.item())]
                        spatial_scores = spatial_distr_sigmoid[spat_dist_score_index]
                        for j, spatial_score in enumerate(spatial_scores):
                            if spatial_score < thresh:
                                continue
                            A[j, i1, i2] = 1
                            
                            if undirected:
                                A[j, i2, i1] = 1 # Symmetric only if undirected

        # Last relationship is not a relationship but just the identity
        A[num_rels-1,:,:] = torch.eye(adj_mat_size)
                
        # Append node features and adjacency matrix
        Xs.append(X)
        Xs_labels.append(X_labels)
        As.append(A)
        
    Xs = torch.stack(Xs)
    As = torch.stack(As)

    return Xs, Xs_labels, As














def construct_dynamic_scene_graph_with_tracking_one_partition(entry_dsg, thresh=0.5, padding=0, num_rels=6):
    """Construct dynamic scene graph from the DSG-Generator output, with one-hot relation augmentation."""
    Xs = []
    As = []

    # Obtain tracklets
    tracklets = entry_dsg['indices']
    tracklets = [t for t in tracklets if t.numel() > 0]

    max_len = max([len(t) for t in tracklets])
    for i, t in enumerate(tracklets):
        if len(t) < max_len:
            padd = torch.full((max_len - len(t),), -1, dtype=t.dtype, device=t.device)
            tracklets[i] = torch.cat((t, padd))

    num_tracklets = len(tracklets)
    len_tracklet = min([len(t) for t in tracklets])

    for frame_idx in range(len_tracklet):
        X = []
        adj_mat_size = max(num_tracklets, padding)
        A = torch.zeros(2, adj_mat_size, adj_mat_size)

        det_id_in_frame = []
        for tr in tracklets:
            det_id_in_frame.append(tr[frame_idx])

        # Construct current adjacency matrix
        pair_idx_list = entry_dsg['pair_idx'].tolist()
        pair_idx_map = {tuple(pair): idx for idx, pair in enumerate(pair_idx_list)}
        spatial_distr_sigmoid = torch.sigmoid(entry_dsg['spatial_distribution'])[0]

        for i1, ele1 in enumerate(det_id_in_frame):
            for i2, ele2 in enumerate(det_id_in_frame):
                if (ele1, ele2) in pair_idx_map:
                    spat_dist_score_index = pair_idx_map[(ele1, ele2)]
                    spatial_scores = spatial_distr_sigmoid[spat_dist_score_index]
                    for j, spatial_score in enumerate(spatial_scores):
                        if spatial_score < thresh:
                            continue
                        A[0, i1, i2] = 1
                        A[0, i2, i1] = 1  # Assume symmetric?

        A[1,:,:] = torch.eye(adj_mat_size)

        # Add current node features
        for idx_in_frame, i in enumerate(det_id_in_frame):
            if i == -1:
                features = torch.zeros_like(entry_dsg['features'][0])  # Padding object
            else:
                features = entry_dsg['features'][i]
            
            # Extend feature with zeros for relationship info
            rel_extension = torch.zeros(num_rels, device=features.device)
            features = torch.cat([features, rel_extension], dim=0)
            X.append(features)

        for _ in range(max(0, padding - num_tracklets)): 
            features = torch.zeros_like(entry_dsg['features'][0])
            features = torch.cat([features, torch.zeros(num_rels, device=features.device)], dim=0)
            X.append(features)

        X = torch.stack(X)

        # Now, for each node, augment its feature vector based on adjacency matrix A
        for i in range(len(det_id_in_frame)):
            for j in range(len(det_id_in_frame)):
                if (det_id_in_frame[i] == -1 or det_id_in_frame[j] == -1):
                    continue
                if (det_id_in_frame[i], det_id_in_frame[j]) in pair_idx_map:
                    spat_dist_score_index = pair_idx_map[(det_id_in_frame[i], det_id_in_frame[j])]
                    spatial_scores = spatial_distr_sigmoid[spat_dist_score_index]
                    for rel_type_idx, spatial_score in enumerate(spatial_scores):
                        if spatial_score >= thresh:
                            X[i][-num_rels + rel_type_idx] = 1  # Set the correct position

        Xs.append(X)
        As.append(A)

    Xs = torch.stack(Xs)
    As = torch.stack(As)

    return Xs, As












def gt_to_tracklet_indices(dsg_gt_annotation, boxes, tracklets):
    """
    Maps ground truth annotations to tracklet indices based on bounding box overlaps.
    Args:
        dsg_gt_annotation (list): A list of ground truth annotations for each frame. Each element is a list where the first element is a dictionary containing frame information, and the remaining elements are dictionaries containing object annotations with 'identifier' and 'bbox' keys.
        boxes (list): A list of detected bounding boxes. Each element is a tensor where the first value is the frame index, and the remaining values are the bounding box coordinates.
        tracklets (list): A list of tracklets. Each tracklet is a list of box indices that belong to the same object across frames.
    Returns:
        list: A list of dictionaries, one for each frame, where keys are object identifiers and values are tracklet indices.
    """
    
    # Extract object identifiers and bounding boxes from the ground truth annotations
    extracted_object_bboxes = {}
    
    for frame_data in dsg_gt_annotation:
        frame_info = frame_data[0]  # First element contains the frame name
        frame_name = frame_info['frame']
        objects = {}
        
        for obj in frame_data[1:]:  # Remaining elements are object annotations
            objects[obj['identifier']] = obj['bbox']
        
        extracted_object_bboxes[frame_name] = objects

    # Compute the intersection over union (IoU) between dsg_gt and detected boxes
    updated_bboxes = {}
    
    for index, (frame, objects) in enumerate(extracted_object_bboxes.items()):
        updated_bboxes[frame] = {}
        
        for identifier, bbox in objects.items():
            best_iou = 0
            best_idx = -1
            
            for i, box in enumerate(boxes):
                if box[0].item() != int(index):
                    continue
                iou = compute_iou(bbox, box[1:].cpu().numpy())
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            updated_bboxes[frame][identifier] = {'bbox': bbox, 'box_id': best_idx}

    # Assign tracklet indices to the updated bounding boxes
    for frame, objects in updated_bboxes.items():
        for identifier, data in objects.items():
            box_id = data['box_id']
            for i, tracklet in enumerate(tracklets):
                if box_id in tracklet:
                    updated_bboxes[frame][identifier]['tracklet_id'] = i
                    break

    # Create output dictionary
    output = []
    for _, objects in updated_bboxes.items():
        curr_frame = {}
        for identifier, data in objects.items():
            if data['box_id'] != -1:
                curr_frame[identifier] = data['tracklet_id']
            else:
                curr_frame[identifier] = -1
        output.append(curr_frame)

    
    return output

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    box1, box2: (x_min, y_min, x_max, y_max)
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    intersection = max(0, xB - xA) * max(0, yB - yA)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    
    return intersection / union if union > 0 else 0




import json
from typing import Any, Optional

def dumps_compact_lists(
    obj: Any,
    indent: int = 4,
    max_inline_len: int = 8,    # soft cap for how many elements we try to inline
    max_width: int = 100        # hard cap for line width when inlining
) -> str:
    """
    Pretty-print dicts with indentation, but keep arrays compact:
    - Inline scalar lists (and lists of small scalar lists) when they fit max_width.
    - Fall back to multiline with indentation otherwise.
    - No spaces inside arrays (e.g., [1,2,[3,4]]) for extra compactness.
    """

    def is_scalar(x: Any) -> bool:
        return isinstance(x, (str, int, float, bool)) or x is None

    def is_scalar_list(x: Any) -> bool:
        return isinstance(x, (list, tuple)) and all(is_scalar(t) for t in x)

    # Try to render an object "inline" (single line). Return None if not feasible.
    def render_inline(o: Any) -> Optional[str]:
        if is_scalar(o):
            return json.dumps(o)

        if is_scalar_list(o):
            # Inline simple 1D lists/tuples of scalars
            parts = [json.dumps(t) for t in o]
            s = '[' + ','.join(parts) + ']'
            return s if len(s) <= max_width else None

        if isinstance(o, (list, tuple)):
            # Inline only if each element is scalar OR a small scalar-list that itself fits
            inlined_items = []
            for e in o:
                if is_scalar(e):
                    s = json.dumps(e)
                elif is_scalar_list(e) and len(e) <= max_inline_len:
                    sub = '[' + ','.join(json.dumps(t) for t in e) + ']'
                    if len(sub) > max_width:
                        return None
                    s = sub
                else:
                    return None
                inlined_items.append(s)

            cand = '[' + ','.join(inlined_items) + ']'
            return cand if len(cand) <= max_width else None

        # Dicts are never inlined at top level
        return None

    def render(o: Any, level: int) -> str:
        sp  = ' ' * (indent * level)
        isp = ' ' * (indent * (level + 1))

        # Try inline first
        inline = render_inline(o)
        if inline is not None:
            return inline

        if isinstance(o, dict):
            if not o:
                return '{}'
            lines = []
            for k, v in o.items():
                val_str = render(v, level + 1)
                lines.append(f'{isp}{json.dumps(k)}: {val_str}')
            return '{\n' + ',\n'.join(lines) + '\n' + sp + '}'

        if isinstance(o, (list, tuple)):
            if not o:
                return '[]'
            elems = []
            for e in o:
                e_inline = render_inline(e)
                if e_inline is not None:
                    elems.append(f'{isp}{e_inline}')
                else:
                    elems.append(f'{isp}{render(e, level + 1)}')
            return '[\n' + ',\n'.join(elems) + '\n' + sp + ']'

        # Scalars
        return json.dumps(o)

    return render(obj, 0)
