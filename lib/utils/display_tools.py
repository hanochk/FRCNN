import numpy as np
import matplotlib.pyplot as plt
from fast_rcnn.bbox_transform import bbox3d_transform_inv
from math_utils import get_3d_box_top_view, bbox_reduce


def display_image_bbox(image, bboxes):
    """Visual debugging of detections."""
    plt.cla()
    plt.imshow(image)
    for i in xrange(np.minimum(500, bboxes.shape[0])):
        bbox = bboxes[i, :4]
        score = bboxes[i, -1]
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='r', linewidth=1)
        )
    plt.show()


def compare_boxes(image, boxes1, boxes2):
    plt.subplot(1, 2, 1)
    display_image_bbox(image, boxes1)
    plt.subplot(1, 2, 2)
    display_image_bbox(image, boxes2)


def verify_targets(blobs, rois, targets, labels):
    im = blobs['data'][0][0, :, :, 0]
    gt_boxes = blobs['gt_boxes']
    fg_idxs = np.where(labels > 0)[0]
    rois = rois[fg_idxs]
    targets = targets[fg_idxs]
    labels = labels[fg_idxs]
    boxes_3d = bbox3d_transform_inv(rois[:, 1:], targets)
    reduced_boxes = bbox_reduce(boxes_3d, labels)
    boxes_2d, _, _ = get_3d_box_top_view(reduced_boxes)
    compare_boxes(im, gt_boxes, boxes_2d)
