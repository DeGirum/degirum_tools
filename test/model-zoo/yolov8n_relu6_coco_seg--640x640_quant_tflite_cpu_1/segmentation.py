import numpy as np
import base64
import json
import math
import cv2


class PostProcessor:
    def __init__(self, json_config):
        self._json_config = json.loads(json_config)
        self._output_conf_threshold = float(
            self._json_config["POST_PROCESS"][0]["OutputConfThreshold"]
        )
        self._output_nms_threshold = float(
            self._json_config["POST_PROCESS"][0]["OutputNMSThreshold"]
        )
        self._maximum_detections = int(
            self._json_config["POST_PROCESS"][0]["MaxDetections"]
        )
        self._output_num_classes = int(
            self._json_config["POST_PROCESS"][0]["OutputNumClasses"]
        )
        self._input_w = int(self._json_config["PRE_PROCESS"][0]["InputW"])
        self._input_h = int(self._json_config["PRE_PROCESS"][0]["InputH"])
        self._input_c = int(self._json_config["PRE_PROCESS"][0]["InputC"])
        self._label_json_config = self._json_config["POST_PROCESS"][0]["LabelsPath"]
        with open(self._label_json_config, "r") as json_file:
            self._labels = json.load(json_file)

    class DFL:
        def __init__(self, c1=16):
            """Integral module of Distribution Focal Loss (DFL)."""
            super().__init__()
            self.c1 = c1

        def softmax(self, x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=1)  # only difference

        def forward(self, x):
            """Applies a transformer layer on numpy array 'x' and returns a numpy array."""
            b, c, a = x.shape  # batch, channels, anchors
            x = x.reshape((b, 4, self.c1, a))
            x = x.transpose(0, 2, 1, 3)
            x = self.softmax(x)
            weights = np.arange(self.c1)
            weights = np.reshape(weights, (1, self.c1, 1, 1))
            output = np.zeros((1, 1, 4, a))
            for i in range(4):
                for j in range(a):
                    output[0, 0, i, j] = np.sum(x[0, :, i, j] * weights[0, :, 0, 0])

            output = output.reshape(b, 4, a)
            return output

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1)  # only difference

    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_arr = [], []
        assert feats is not None
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = np.arange(0, w, dtype=float) + grid_cell_offset
            sy = np.arange(0, h, dtype=float) + grid_cell_offset
            sy, sx = np.meshgrid(sy, sx)
            anchor_points.append(np.stack((sy, sx), axis=-1).reshape((-1, 2)))
            stride_arr.append(np.full((h * w, 1), stride, dtype=float))
        return np.concatenate(anchor_points), np.concatenate(stride_arr)

    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = np.split(distance, 2, axis=1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return np.concatenate((c_xy, wh), dim)
        return np.concatenate((c_xy, wh), dim)  # xyxy bbox

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def decode_bbox(self, preds, img_shape):
        """A list of predictions are decoded to shape [1, 84, 8400]"""
        num_classes = next((o.shape[2] for o in preds if o.shape[2] != 64), -1)
        assert (
            num_classes != -1
        ), "cannot infer postprocessor inputs via output shape if there are 64 classes"
        pos = [
            i
            for i, _ in sorted(
                enumerate(preds),
                key=lambda x: (
                    x[1].shape[2] if num_classes > 64 else -x[1].shape[2],
                    -x[1].shape[1],
                ),
            )
        ]
        x = np.concatenate(
            [
                np.concatenate([preds[i] for i in pos[: len(pos) // 2]], 1),
                np.concatenate([preds[i] for i in pos[len(pos) // 2 :]], 1),
            ],
            2,
        )
        x = np.transpose(x, (0, 2, 1))
        reg_max = (x.shape[1] - num_classes) // 4
        dfl = self.DFL(reg_max)
        img_h, img_w = img_shape[-2], img_shape[-1]
        strides = [
            int(math.sqrt(img_shape[-2] * img_shape[-1] / preds[p].shape[1]))
            for p in pos
            if preds[p].shape[2] != 64
        ]
        dims = [(img_h // s, img_w // s) for s in strides]
        fake_feats = [np.zeros((1, 1, h, w)) for h, w in dims]
        anchors, strides = (
            x.transpose(1, 0) for x in self.make_anchors(fake_feats, strides, 0.5)
        )  # generate anchors and strides
        box = x[:, :-num_classes, :]
        dbox = (
            self.dist2bbox(
                dfl.forward(box) if reg_max > 1 else box,
                np.expand_dims(anchors, axis=0),
                xywh=True,
                dim=1,
            )
            * strides
        )
        cls = x[:, -num_classes:, :]
        y = np.concatenate((dbox, self.sigmoid(cls)), 1)
        return y

    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.
        """
        y = np.copy(x)
        y[..., 0:2] = x[..., 0:2] - x[..., 2:4] / 2  # top left (x, y)
        y[..., 2:4] = x[..., 0:2] + x[..., 2:4] / 2  # bottom right (x, y)
        return y

    def nms(self, boxes, overlap_threshold=0.2, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        index_array = scores.argsort()[::-1]
        keep = []
        while index_array.size > 0:
            keep.append(index_array[0])
            x1_ = np.maximum(x1[index_array[0]], x1[index_array[1:]])
            y1_ = np.maximum(y1[index_array[0]], y1[index_array[1:]])
            x2_ = np.minimum(x2[index_array[0]], x2[index_array[1:]])
            y2_ = np.minimum(y2[index_array[0]], y2[index_array[1:]])

            w = np.maximum(0.0, x2_ - x1_)
            h = np.maximum(0.0, y2_ - y1_)
            inter = w * h

            if min_mode:
                overlap = inter / np.minimum(
                    areas[index_array[0]], areas[index_array[1:]]
                )
            else:
                overlap = inter / (
                    areas[index_array[0]] + areas[index_array[1:]] - inter
                )

            inds = np.where(overlap <= overlap_threshold)[0]
            index_array = index_array[inds + 1]
        return keep

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.8,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=80,  # number of classes (optional)
        max_nms=30000,
        max_wh=7680,
    ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (np.ndarray): A numpy array of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, numpy array]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[numpy array]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """
        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_thres <= 1
        ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(
            prediction, (list, tuple)
        ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        output = []
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates
        # Settings
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = np.transpose(prediction, (0, 2, 1))
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy

        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + nm + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = np.concatenate((x, v), axis=0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            #         # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x[:, :4], x[:, 4 : nc + 4], x[:, nc + 4 :]
            if multi_label:
                i, j = np.where(cls > conf_thres)
                x = np.concatenate(
                    (box[i], x[i, 4 + j, None], j[:, None], mask[i]), axis=1
                )
            else:  # best class only
                conf = np.max(cls, axis=1, keepdims=True)
                j = np.argmax(cls[:, :], axis=1)  # Remove keepdims
                j = np.expand_dims(j, axis=1)  # Add the dimension back
                x = np.concatenate((box, conf, j, mask), axis=1)

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == np.any(classes, axis=1))]

            #       # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[
                    x[:, 4].argsort()[::-1][:max_nms]
                ]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

            scores = scores.reshape(scores.shape[0], 1)
            con = np.concatenate((boxes, scores), axis=1)
            keep_boxes = self.nms(con, iou_thres)  # NMS
            keep_boxes = keep_boxes[:max_det]  # limit detections

            for k in keep_boxes:
                output.append(np.array([x[k]]))
        return output

    def crop_mask(self, masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1))
        r = np.arange(w)[None, None, :]  # rows shape(1,1,w)
        c = np.arange(h)[None, :, None]  # cols shape(1,h,1)
        return np.where(((r >= x1) & (r < x2) & (c >= y1) & (c < y2)), masks, 0)

    def process_mask(self, protos, masks_in, bboxes, shape):
        """
        Scale masks to model input, and crop them using bounding boxes.
        """
        c, mh, mw = protos.shape
        protos = protos.astype(float)
        protos = protos.reshape(c, (protos.shape[1] * protos.shape[2]))
        n_shape = (masks_in @ protos).shape
        masks = self.sigmoid(masks_in @ protos).reshape(n_shape[0], mh, mw)
        masks = np.moveaxis(masks, 0, -1)  # masks, (H, W, N)
        masks = self.scale_image(masks, shape)
        masks = np.moveaxis(masks, -1, 0)  # masks, (N, H, W))
        masks = self.crop_mask(masks, bboxes)  # Crop masks using bounding boxes (CHW)
        mask_ = np.where(masks > 0.5, 1, 0)
        return mask_.astype(float)

    def scale_image(self, masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size
        """
        # Rescale coordinates (xyxy) from im1_shape to im0_shape
        im1_shape = masks.shape
        if im1_shape[:2] == im0_shape[:2]:
            return masks
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(
                im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1]
            )  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (
                im1_shape[0] - im0_shape[0] * gain
            ) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
        if len(masks.shape) < 2:
            raise ValueError(
                f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
            )
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
        if len(masks.shape) == 2:
            masks = masks[:, :, None]

        return masks

    def run_length_encode(self, x):
        """Returns run length encoded string for data"""
        array_shape = x.shape
        x = x.flatten()
        starts = np.r_[0, np.flatnonzero(np.diff(x)) + 1]
        lengths = np.diff(np.r_[starts, x.size]).astype(np.uint32)
        values = x[starts].astype(np.uint32)

        rle = np.concatenate((values, lengths))
        res = {
            "height": array_shape[0],
            "width": array_shape[1],
            "data": base64.b64encode(rle.tobytes()).decode("ascii"),
        }
        return res

    def forward(self, tensor_list, details_list):
        new_inference_results = []
        float_preds_list = []
        mcv = float("-inf")
        lci = -1
        pidx = -1

        for idx, s in enumerate(tensor_list):
            qp = details_list[idx]["quantization_parameters"]
            de_quantization_zero_parameter = float(qp["zero_points"][0])
            de_quantization_scale_parameter = float(qp["scales"][0])
            s = s.astype(np.float32)
            s = (s - de_quantization_zero_parameter) * de_quantization_scale_parameter
            float_preds_list.append(s)
            dim_1 = s.shape[1]
            if dim_1 > mcv:
                mcv = dim_1
                lci = idx
            if len(s.shape) == 4:
                pidx = idx

        pred_order = [
            item
            for index, item in enumerate(float_preds_list)
            if index not in [pidx, lci]
        ]
        pred_decoded = self.decode_bbox(
            pred_order, (1, self._input_c, self._input_w, self._input_h)
        )
        mask = float_preds_list[lci]
        proto = float_preds_list[pidx]
        proto = proto.transpose(0, 3, 1, 2)
        pred_decoded = np.concatenate([pred_decoded, np.transpose(mask, (0, 2, 1))], 1)

        p = self.non_max_suppression(
            pred_decoded,
            conf_thres=self._output_conf_threshold,
            iou_thres=self._output_nms_threshold,
            agnostic=False,
            multi_label=True,
            max_det=self._maximum_detections,
            classes=None,
            nc=self._output_num_classes,
        )
        if len(p) > 0:
            pred = np.vstack(p)
            masks = self.process_mask(
                proto[0], pred[:, 6:], pred[:, :4], (self._input_h, self._input_w)
            )  # HW

            for i in range(len(pred)):
                result = {
                    "bbox": pred[i][:4].tolist(),
                    "category_id": int(pred[i][5]),
                    "label": self._labels[str(int(pred[i][5]))],
                    "score": float(pred[i][4]),
                    "mask": self.run_length_encode(masks[i]),
                }
                new_inference_results.append(result)

        return json.dumps(new_inference_results)
