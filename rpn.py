import numpy as np
import math
import megengine
import megengine.functional as F
import megengine.module as M
from bounding_box import BoxList

def boxlist_iou(boxlist1, boxlist2):
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    N = len(boxlist1)
    M = len(boxlist2)
    area1 = boxlist1.area()
    area2 = boxlist2.area()
    box1, box2 = boxlist1.bbox, boxlist2.bbox
    lt = F.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = F.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    TO_REMOVE = 1
    wh = F.clip((rb - lt + TO_REMOVE), lower=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def _cat(tensors, dim=0):
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return F.concat(tensors, dim)

def cat_boxlist(bboxes):
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def boxlist_ml_nms(boxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    return boxlist.convert(mode)

def remove_small_boxes(boxlist, min_size):
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = F.split(xywh_boxes, 4, axis=-1)
    keep = megengine.tensor(((ws >= min_size) & (hs >= min_size)).numpy().nonzero()[0])
    return boxlist[keep]

class BufferList(M.Module):

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

    def forward(self):
        pass

class BoxCoder():
    def __init__(self):
        pass
    def decode(self, preds, anchors):
        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = preds[:, 2::4] / ww
        dh = preds[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = F.clip(dw, upper=math.log(1000. / 16))
        dh = F.clip(dh, upper=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = F.exp(dw) * widths[:, None]
        pred_h = F.exp(dh) * heights[:, None]

        pred_boxes = F.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes

def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )

def _generate_anchors(base_size, scales, aspect_ratios):
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 0.5
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return megengine.Tensor(anchors)

def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

class AnchorGenerator(M.Module):
    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                )
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        self.strides = anchor_strides
        self.cell_anchors = cell_anchors
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            grid_height, grid_width = size
            shifts_x = F.arange(0, grid_width * stride, step=stride, dtype=np.float32)
            shifts_y = F.arange(0, grid_height * stride, step=stride, dtype=np.float32)
            shift_x, shift_y = np.meshgrid(shifts_x.numpy(), shifts_y.numpy())
            shift_x = megengine.tensor(shift_x)
            shift_y = megengine.tensor(shift_y)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = F.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

            anchors.append(
                (shifts.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4)
            )
        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
       
        inds_inside = (
            (anchors[..., 0] >= -self.straddle_thresh)
            & (anchors[..., 1] >= -self.straddle_thresh)
            & (anchors[..., 2] < image_width + self.straddle_thresh)
            & (anchors[..., 3] < image_height + self.straddle_thresh)
        )
       
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors
        
def make_anchor_generator_paa():
    anchor_sizes = (64, 128, 256, 512, 1024)
    aspect_ratios = (1.0,)
    anchor_strides = (8, 16, 32, 64, 128)
    straddle_thresh = 0
    octave = 2.0
    scales_per_octave = 1

    assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

    anchor_generator = AnchorGenerator(
        tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh
    )
    return anchor_generator

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.reshape(N, -1, C, H, W)
    layer = layer.transpose(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

class PAAPostProcessor(M.Module):
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        box_coder,
        bbox_aug_enabled=False,
        bbox_aug_vote=False,
        score_voting=False,
    ):
        super(PAAPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
        self.box_coder = box_coder
        self.bbox_aug_vote = bbox_aug_vote
        self.score_voting = score_voting

    def forward_for_single_feature_map(self, box_cls, box_regression, iou_pred, anchors):
        N, _, H, W = box_cls.shape
        A = box_regression.shape[1] // 4
        C = box_cls.shape[1] // A
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = F.sigmoid(box_cls)
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = F.clip(pre_nms_top_n, upper=self.pre_nms_top_n)
        if iou_pred is not None:
            iou_pred = permute_and_flatten(iou_pred, N, A, 1, H, W)
            iou_pred = F.sigmoid(iou_pred.reshape(N, -1))
            box_cls = F.sqrt((box_cls * iou_pred[:, :, None]))
        results = []
        for per_box_cls_, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
            per_box_cls = per_box_cls_[per_candidate_inds]
            if len(per_box_cls) == 0:
                continue
            per_box_cls, top_k_indices = F.topk(per_box_cls, per_pre_nms_top_n, descending=True, no_sort=True)
            s0, s1 = per_candidate_inds.numpy().nonzero()
            per_candidate_nonzeros = F.stack([megengine.tensor(s0), megengine.tensor(s1)], axis=-1)[top_k_indices, :]
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].reshape(-1, 4),
                per_anchors.bbox[per_box_loc, :].reshape(-1, 4)
            )
            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, box_cls, box_regression, iou_pred, anchors):
        sampled_boxes = []
        anchors = list(zip(*anchors))
        if iou_pred is None:
            iou_pred = [None] * len(box_cls)
        for _, (o, b, i, a) in enumerate(zip(box_cls, box_regression, iou_pred, anchors)):
            sampled_boxes.append(self.forward_for_single_feature_map(o, b, i, a))
        boxlists = [[]]
        for i in range(5):
            if len(sampled_boxes[i]) != 0:
                boxlists[0].append(sampled_boxes[i][0])

        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
            boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh = F.sort(cls_scores)[0][number_of_detections - self.fpn_post_nms_top_n]
                keep = cls_scores >= image_thresh.item()
                keep = megengine.tensor(keep.numpy().nonzero()[0])
                result = result[keep]
            if self.score_voting:
                boxes_al = boxlists[i].bbox
                boxlist = boxlists[i]
                labels = boxlists[i].get_field("labels")
                scores = boxlists[i].get_field("scores")
                sigma = 0.025
                result_labels = result.get_field("labels")
                for j in range(1, self.num_classes):
                    inds = megengine.tensor((labels == j).numpy().nonzero()[0])
                    scores_j = scores[inds]
                    boxes_j = boxes_al[inds, :].reshape(-1, 4)
                    boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                    result_inds = megengine.tensor((result_labels == j).numpy().nonzero()[0])

                    boxlist_for_class_nmsed = result[result_inds]
                    ious = boxlist_iou(boxlist_for_class_nmsed, boxlist_for_class)
                    voted_boxes = []
                    for bi in range(len(boxlist_for_class_nmsed)):
                        cur_ious = ious[bi]
                        pos_inds = megengine.tensor((cur_ious > 0.01).numpy().nonzero()[0])
                        pos_ious = cur_ious[pos_inds]
                        pos_boxes = boxlist_for_class.bbox[pos_inds]
                        pos_scores = scores_j[pos_inds]
                        pis = (F.exp(-(1-pos_ious)**2 / sigma) * pos_scores).reshape(-1, 1)
                        voted_box = F.sum(pos_boxes * pis, axis=0) / F.sum(pis, axis=0)
                        voted_boxes.append(voted_box.reshape(1,-1))
                    if voted_boxes:
                        voted_boxes = F.concat(voted_boxes, axis=0)
                        boxlist_for_class_nmsed_ = BoxList(
                            voted_boxes,
                            boxlist_for_class_nmsed.size,
                            mode="xyxy")
                        boxlist_for_class_nmsed_.add_field(
                            "scores",
                            boxlist_for_class_nmsed.get_field('scores'))
                        result.bbox[result_inds] = boxlist_for_class_nmsed_.bbox
            results.append(result)
        return results

def make_paa_postprocessor(box_coder):

    box_selector = PAAPostProcessor(
        pre_nms_thresh=0.05,
        pre_nms_top_n=1000,
        nms_thresh=0.6,
        fpn_post_nms_top_n=100,
        min_size=0,
        num_classes=81,
        bbox_aug_enabled=False,
        box_coder=box_coder,
        bbox_aug_vote=False,
        score_voting=True,
    )
    return box_selector

class Scale(M.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = megengine.Parameter(megengine.tensor([init_value], dtype=np.float32))
    def forward(self, input):
        return input * self.scale

class PAAHead(M.Module):
    def __init__(self, in_channels):
        super(PAAHead, self).__init__()
        num_classes = 80
        num_anchors = 1
        self.use_iou_pred = True
        cls_tower = []
        bbox_tower = []
        for i in range(4):
            conv_func = M.Conv2d
            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(M.GroupNorm(32, in_channels))
            cls_tower.append(M.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(M.GroupNorm(32, in_channels))
            bbox_tower.append(M.ReLU())
        setattr(self, 'cls_tower', M.Sequential(*cls_tower))
        setattr(self, 'bbox_tower', M.Sequential(*bbox_tower))
        self.cls_logits = M.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = M.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )
        all_modules = [self.cls_tower, self.bbox_tower,
                       self.cls_logits, self.bbox_pred]
        if self.use_iou_pred:
            self.iou_pred = M.Conv2d(
                in_channels, num_anchors * 1, kernel_size=3, stride=1,
                padding=1
            )
            all_modules.append(self.iou_pred)

        # initialize the bias for focal loss
        prior_prob = 0.01
        self.scales = list([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        iou_pred = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_reg.append(bbox_pred)
            if self.use_iou_pred:
                iou_pred.append(self.iou_pred(box_tower))
        res = [logits, bbox_reg]
        if self.use_iou_pred:
            res.append(iou_pred)
        return res 

class PAAModule(M.Module):
    def __init__(self, in_channels):
        super(PAAModule, self).__init__()
        self.head = PAAHead(in_channels)
        box_coder = BoxCoder()
        self.box_selector_test = make_paa_postprocessor(box_coder)
        self.anchor_generator = make_anchor_generator_paa()
        self.use_iou_pred = True

    def forward(self, images, features, targets=None):
        preds = self.head(features)
        box_cls, box_regression = preds[:2]
        iou_pred = preds[2] if self.use_iou_pred else None
        anchors = self.anchor_generator(images, features)
        return self._forward_test(box_cls, box_regression, iou_pred, anchors)

    def _forward_test(self, box_cls, box_regression, iou_pred, anchors):
        boxes = self.box_selector_test(box_cls, box_regression, iou_pred, anchors)
        return boxes, {}