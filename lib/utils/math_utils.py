import numpy as np
from fast_rcnn.bbox_transform import bbox3d_transform_inv

def rotation_mat_z(angle, dim):
    """Clockwise rotation around z axis in a left handed coordinate system. Images are right handed systems and thus 
    counter clockwise there"""
    if dim == '2d':
        return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    elif dim == '3d':
        return np.array([[np.cos(angle), np.sin(angle), 0.], [-np.sin(angle), np.cos(angle), 0.], 0., 0., 1.])
    else:
        raise NotImplementedError('math_utils.py:rotation_mat support only 2d or 3d input')


def real_to_image_conversion(bboxes, ranges, resolutions):
    """ Takes an nd array of bounding boxes in minimum point maximum point format, in real world coordinates and 
    converts it to image coordinates."""
    bboxes = np.atleast_2d(bboxes)
    ranges = ranges[:, :2]
    zero_point = ranges[0]
    max_point = ranges[1]
    first_point = resolutions[:2]
    # clip boxes with minimal point before zero point or maximal point after max point
    bboxes[bboxes[:, 0] < zero_point[0]] = zero_point[0]
    bboxes[bboxes[:, 1] < zero_point[1]] = zero_point[1]
    bboxes[bboxes[:, 2] > max_point[0]] = max_point[0]
    bboxes[bboxes[:, 3] > max_point[1]] = max_point[1]

    # remove boxes with minimal point after max point or maximal point before zero point
    keep_inds = np.where((bboxes[:, 0] < max_point[0]) &
                         (bboxes[:, 1] < max_point[1]) &
                         (bboxes[:, 2] > zero_point[0]) &
                         (bboxes[:, 3] > zero_point[1]))[0]
    bboxes = bboxes[keep_inds] - np.tile(zero_point, 2)[np.newaxis, :]
    bboxes /= np.tile(first_point, 2)[np.newaxis, :]
    bboxes[:, np.arange(4)] = bboxes[:, np.array([1, 0, 3, 2])]
    return bboxes, keep_inds


def get_3d_box_top_view(box_3d, ranges=np.array([[0, -25.6, -2.2], [102.4, 25.6, 1.]]),
                        resolutions=np.array([0.2, 0.2, 0.4])):
    box_3d = np.atleast_2d(box_3d)
    theta = box_3d[:, 6]  # rotation angle around z axis
    corners_offset = 0.5 * np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float)
    center = box_3d[:, :2]
    corners = center[:, np.newaxis, :] + \
              np.matmul(corners_offset[np.newaxis, ...] * box_3d[:, 3:5][:, np.newaxis, :], rotation_mat_z(theta, '2d').transpose(2, 0, 1))

    min_point = corners.min(axis=1)
    max_point = corners.max(axis=1)
    box_2d = np.hstack((min_point, max_point))
    # TODO: create a single im_info containing all relevant info about image
    box_2d_image, keep_inds = real_to_image_conversion(box_2d, ranges, resolutions)
    return box_2d_image, keep_inds, theta


def bbox_reduce(boxes_3d, labels):
    reduced_boxes = boxes_3d[np.arange(boxes_3d.shape[0])[:, np.newaxis], np.arange(7)[np.newaxis, :] + 7 * labels]
    return reduced_boxes


def get_2d_boxes_from_deltas(rois, bbox_deltas, labels):
    boxes_3d = bbox3d_transform_inv(rois, bbox_deltas)
    reduced_boxes = bbox_reduce(boxes_3d, labels)
    boxes_2d = get_3d_box_top_view(reduced_boxes)

    return boxes_2d