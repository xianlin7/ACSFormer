import numpy as np
import torch

def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    The decoding formulas are used in works such as R-CNN [#]_.

    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin}, \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

    """

    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc

def prior2loc(prior_bbox, dst_bbox):
    """
    prior_bbox :[y x h w]
    """

    height = prior_bbox[:, 2]
    width = prior_bbox[:, 3]
    ctr_y = prior_bbox[:, 0]
    ctr_x = prior_bbox[:, 1]

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = 1
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log((base_height + 1) / height)
    dw = np.log((base_width + 1) / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc

def obtain_gt_loc(gt_bbox, gt_label, numclass, downsample=True):
    loc_normalize_mean = (0., 0., 0., 0.),
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    gt_class_loc = np.zeros((numclass, 4), dtype=np.float32)
    gt_class_label = np.zeros((numclass))

    if gt_bbox.shape[0] == 0:
        return gt_class_loc, gt_class_label

    prior_bbox = np.array([[303, 380, 95, 86], [305, 171, 59, 58], [308, 346, 56, 60], [211, 167, 39, 36],
                           [249, 170, 160, 161], [218, 310, 90, 97], [277, 270, 31, 29], [239, 269, 49, 79]]) # [y x h w]
    if downsample:
        prior_bbox = prior_bbox//2
        gt_bbox = gt_bbox//2
    
    related_bbox = prior_bbox[gt_label-1, :]
    gt_loc = prior2loc(related_bbox, gt_bbox)
    gt_loc = ((gt_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

    gt_class_loc[gt_label, :] = gt_loc
    gt_class_label[gt_label] = 1
    gt_class_label[0] =1

    return gt_class_loc, gt_class_label

def obtain_gt_loc_acdc(gt_bbox, gt_label, numclass, downsample=False):
    loc_normalize_mean = (0., 0., 0., 0.),
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    gt_class_loc = np.zeros((numclass, 4), dtype=np.float32)
    gt_class_label = np.zeros((numclass))

    if gt_bbox.shape[0] == 0:
        return gt_class_loc, gt_class_label

    prior_bbox = np.array([[100, 130, 38, 42], [122, 126, 48, 43], [122, 126, 38, 30]]) # [y x h w]
    if downsample:
        prior_bbox = prior_bbox//2
        gt_bbox = gt_bbox//2
    
    related_bbox = prior_bbox[gt_label-1, :]
    gt_loc = prior2loc(related_bbox, gt_bbox)
    gt_loc = ((gt_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

    gt_class_loc[gt_label, :] = gt_loc
    gt_class_label[gt_label] = 1
    gt_class_label[0] =1

    return gt_class_loc, gt_class_label

def obtain_gt_loc_prostate(gt_bbox, gt_label, numclass, downsample=False):
    loc_normalize_mean = (0., 0., 0., 0.),
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    gt_class_loc = np.zeros((numclass, 4), dtype=np.float32)
    gt_class_label = np.zeros((numclass))

    if gt_bbox.shape[0] == 0:
        return gt_class_loc, gt_class_label

    prior_bbox = np.array([[133, 127, 32, 56], [121, 127, 40, 49]]) # [y x h w]
    if downsample:
        prior_bbox = prior_bbox//2
        gt_bbox = gt_bbox//2
    
    related_bbox = prior_bbox[gt_label-1, :]
    gt_loc = prior2loc(related_bbox, gt_bbox)
    gt_loc = ((gt_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

    gt_class_loc[gt_label, :] = gt_loc
    gt_class_label[gt_label] = 1
    gt_class_label[0] =1

    return gt_class_loc, gt_class_label

def obtain_gt_loc_lits(gt_bbox, gt_label, numclass, downsample=False):
    loc_normalize_mean = (0., 0., 0., 0.),
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    gt_class_loc = np.zeros((numclass, 4), dtype=np.float32)
    gt_class_label = np.zeros((numclass))

    if gt_bbox.shape[0] == 0:
        return gt_class_loc, gt_class_label

    prior_bbox = np.array([[129, 86, 87, 83], [132, 84, 44, 42]]) # [y x h w]
    if downsample:
        prior_bbox = prior_bbox//2
        gt_bbox = gt_bbox//2
    
    related_bbox = prior_bbox[gt_label-1, :]
    gt_loc = prior2loc(related_bbox, gt_bbox)
    gt_loc = ((gt_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

    gt_class_loc[gt_label, :] = gt_loc
    gt_class_label[gt_label] = 1
    gt_class_label[0] =1

    return gt_class_loc, gt_class_label

def obtain_gt_loc_ich(gt_bbox, gt_label, numclass, downsample=False):
    loc_normalize_mean = (0., 0., 0., 0.),
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    gt_class_loc = np.zeros((numclass, 4), dtype=np.float32)
    gt_class_label = np.zeros((numclass))

    if gt_bbox.shape[0] == 0:
        return gt_class_loc, gt_class_label

    prior_bbox = np.array([[135, 124, 41, 35]]) # [y x h w]
    if downsample:
        prior_bbox = prior_bbox//2
        gt_bbox = gt_bbox//2
    
    related_bbox = prior_bbox[gt_label-1, :]
    gt_loc = prior2loc(related_bbox, gt_bbox)
    gt_loc = ((gt_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

    gt_class_loc[gt_label, :] = gt_loc
    gt_class_label[gt_label] = 1
    gt_class_label[0] =1

    return gt_class_loc, gt_class_label

def obtain_gt_loc_chaos(gt_bbox, gt_label, numclass, downsample=False):
    loc_normalize_mean = (0., 0., 0., 0.),
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    gt_class_loc = np.zeros((numclass, 4), dtype=np.float32)
    gt_class_label = np.zeros((numclass))
    gt_class_label[0] =1

    if gt_bbox.shape[0] == 0:
        return gt_class_loc, gt_class_label

    prior_bbox = np.array([[123, 86, 90, 86]]) # [y x h w]
    if downsample:
        prior_bbox = prior_bbox//2
        gt_bbox = gt_bbox//2
    
    related_bbox = prior_bbox[gt_label-1, :]
    gt_loc = prior2loc(related_bbox, gt_bbox)
    gt_loc = ((gt_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

    gt_class_loc[gt_label, :] = gt_loc
    gt_class_label[gt_label] = 1

    return gt_class_loc, gt_class_label


def obtain_gt_loc_task09spleen(gt_bbox, gt_label, numclass, downsample=False):
    loc_normalize_mean = (0., 0., 0., 0.),
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    gt_class_loc = np.zeros((numclass, 4), dtype=np.float32)
    gt_class_label = np.zeros((numclass))
    gt_class_label[0] =1

    if gt_bbox.shape[0] == 0:
        return gt_class_loc, gt_class_label

    prior_bbox = np.array([[106, 64, 43, 39]]) # [y x h w]
    if downsample:
        prior_bbox = prior_bbox//2
        gt_bbox = gt_bbox//2
    
    related_bbox = prior_bbox[gt_label-1, :]
    gt_loc = prior2loc(related_bbox, gt_bbox)
    gt_loc = ((gt_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

    gt_class_loc[gt_label, :] = gt_loc
    gt_class_label[gt_label] = 1

    return gt_class_loc, gt_class_label



def obtain_gt_loc_MosMed(gt_bbox, gt_label, numclass, downsample=False):
    loc_normalize_mean = (0., 0., 0., 0.),
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    gt_class_loc = np.zeros((numclass, 4), dtype=np.float32)
    gt_class_label = np.zeros((numclass))
    gt_class_label[0] =1

    if gt_bbox.shape[0] == 0:
        return gt_class_loc, gt_class_label

    prior_bbox = np.array([[152, 128, 38, 66]]) # [y x h w]
    if downsample:
        prior_bbox = prior_bbox//2
        gt_bbox = gt_bbox//2
    
    related_bbox = prior_bbox[gt_label-1, :]
    gt_loc = prior2loc(related_bbox, gt_bbox)
    gt_loc = ((gt_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

    gt_class_loc[gt_label, :] = gt_loc
    gt_class_label[gt_label] = 1

    return gt_class_loc, gt_class_label


def prior2bbox(loc, downsample=True):
    prior_bbox = np.array([[[1, 1, 1, 1], [303, 380, 95, 86], [305, 171, 59, 58], [308, 346, 56, 60], [211, 167, 39, 36],
                           [249, 170, 160, 161], [218, 310, 90, 97], [277, 270, 31, 29], [239, 269, 49, 79]]]) # [y x h w]
    if downsample:
        prior_bbox = prior_bbox//2
        
    loc_normalize_mean = [[0., 0., 0., 0.]]
    loc_normalize_std = [[0.1, 0.1, 0.2, 0.2]]

    mean = torch.Tensor(loc_normalize_mean).cuda().repeat(loc.shape[1], 1)
    std = torch.Tensor(loc_normalize_std).cuda().repeat(loc.shape[1], 1)

    loc = (loc * std) + mean
    prior_bbox = torch.Tensor(prior_bbox).repeat(loc.shape[0], 1, 1)
    loc = tonumpy(loc)
    prior_bbox = tonumpy(prior_bbox)

    prior_height = prior_bbox[:, :, 2]
    prior_width = prior_bbox[:, :, 3]
    prior_ctr_y = prior_bbox[:, :, 0]
    prior_ctr_x = prior_bbox[:, :, 1]

    dy = loc[:, :, 0] # b class 4
    dx = loc[:, :, 1]
    dh = loc[:, :, 2]
    dw = loc[:, :, 3]

    ctr_y = dy * prior_height + prior_ctr_y
    ctr_x = dx * prior_width + prior_ctr_x
    h = np.exp(dh) * prior_height
    w = np.exp(dw) * prior_width

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, :, 0] = ctr_y - 0.5 * h
    dst_bbox[:, :, 1] = ctr_x - 0.5 * w
    dst_bbox[:, :, 2] = ctr_y + 0.5 * h
    dst_bbox[:, :, 3] = ctr_x + 0.5 * w

    return dst_bbox


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()    


def smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def loc_loss(pred_loc, gt_loc, gt_label, sigma=1):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0), :] = 1
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    bbox_num = (gt_label >= 0).sum().float()
    if bbox_num > 0:
        loc_loss /= (bbox_num) # ignore gt_label==-1 for rpn_loss
    else:
        loc_loss = 0
    return loc_loss
    

                  
    