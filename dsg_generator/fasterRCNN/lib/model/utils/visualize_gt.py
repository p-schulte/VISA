import cv2
import numpy as np
import torch

def visualize_gt(im_data, gt_boxes):
    """
    Visualize ground-truth bounding boxes on the image.
    
    Args:
        im_data (torch.Tensor): Image tensor of shape (1, C, H, W)
        gt_boxes (torch.Tensor): Ground truth bounding boxes (x1, y1, x2, y2, class)
    """

    # Convert im_data (Tensor) to NumPy image
    im_data_np = im_data[0].cpu().numpy().transpose(1, 2, 0)  # (C, H, W)

    # Check if image is normalized (assumed float [0,1] or [0,255])
    if im_data_np.max() > 1.0:
        im_data_np = im_data_np.astype(np.uint8)  # Already [0,255]
    else:
        im_data_np = (im_data_np * 255).astype(np.uint8)  # Convert to [0,255]

    # Ensure values are within valid range
    im_data_np = np.clip(im_data_np, 0, 255)

    # Convert RGB (PyTorch) to BGR (OpenCV)
    im_data_np = cv2.cvtColor(im_data_np, cv2.COLOR_RGB2BGR)

    # Copy image for visualization
    im2show = np.copy(im_data_np)

    # Load ground-truth bounding boxes
    gt_boxes_np = gt_boxes[0].cpu().numpy().squeeze()  # Convert to NumPy
    num_gt_boxes = gt_boxes_np.shape[0]  # Number of GT boxes

    for j in range(num_gt_boxes):
        x1, y1, x2, y2 = gt_boxes_np[j, :4]  # Extract bbox
        label = int(gt_boxes_np[j, 4]) if gt_boxes_np.shape[1] > 4 else "GT"  # Class index (if exists)

        # Draw GT bounding boxes in **blue** (255, 0, 0) for consistency
        cv2.rectangle(im2show, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(im2show, f"GT-{label}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save and show the image
    cv2.imwrite('gt_visualization.png', im2show)
