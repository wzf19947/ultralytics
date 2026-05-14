import axengine as axe
import cv2
import numpy as np
import time
import glob
import os

names = [
    "Cow"
]

def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """Perform non-maximum suppression (NMS) on prediction results using NumPy only."""
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    
    # Convert to numpy if needed
    if not isinstance(prediction, np.ndarray):
        prediction = np.asarray(prediction)
    
    if classes is not None:
        classes = np.asarray(classes)
    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = []
        for pred in prediction:
            mask = pred[:, 4] > conf_thres
            filtered = pred[mask][:max_det]
            if classes is not None:
                class_mask = np.any(filtered[:, 5:6] == classes, axis=1)
                filtered = filtered[class_mask]
            output.append(filtered)
        return output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates
    
    # Create index arrays
    xinds = np.arange(prediction.shape[-1], dtype=np.int32)
    xinds_expanded = np.tile(xinds[np.newaxis, :, np.newaxis], (bs, 1, 1))

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = np.transpose(prediction, (0, 2, 1))  # shape(1,6300,84)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = []
    keepi = []
    
    for xi in range(bs):
        x = prediction[xi]
        xk = xinds_expanded[xi]
        
        # Apply confidence threshold
        filt = xc[xi]
        x = x[filt]
        xk_filtered = xk[filt]

        if x.shape[0] == 0:
            output.append(np.zeros((0, 6 + extra), dtype=np.float32))
            keepi.append(np.zeros((0, 1), dtype=np.int32))
            continue

        # Split boxes and classes
        box = x[:, :4]
        cls = x[:, 4:4+nc]
        mask = x[:, 4+nc:] if extra > 0 else np.empty((x.shape[0], 0))

        if multi_label:
            i, j = np.where(cls > conf_thres)
            selected_box = box[i]
            selected_conf = cls[i, j:j+1]
            selected_j = j[:, np.newaxis]
            selected_mask = mask[i]
            x = np.concatenate([selected_box, selected_conf, selected_j.astype(np.float32), selected_mask], axis=1)
            xk_filtered = xk_filtered[i]
        else:
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            filt = conf[:, 0] > conf_thres
            x = np.concatenate([box, conf, j.astype(np.float32), mask], axis=1)[filt]
            xk_filtered = xk_filtered[filt]

        # Filter by class
        if classes is not None:
            class_mask = np.any(x[:, 5:6] == classes, axis=1)
            x = x[class_mask]
            xk_filtered = xk_filtered[class_mask]

        n = x.shape[0]
        if n == 0:
            output.append(np.zeros((0, 6 + extra), dtype=np.float32))
            keepi.append(np.zeros((0, 1), dtype=np.int32))
            continue
        
        if n > max_nms:
            sorted_idx = np.argsort(-x[:, 4])[:max_nms]
            x = x[sorted_idx]
            xk_filtered = xk_filtered[sorted_idx]

        # NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        
        if not rotated:
            boxes = x[:, :4] + c
            i = numpy_nms(boxes, scores, iou_thres)
        else:
            boxes = np.concatenate([x[:, :2] + c, x[:, 2:4], x[:, -1:]], axis=-1)
            i = numpy_nms(boxes[:, :4], scores, iou_thres)  # Simplified for rotated boxes
        
        i = i[:max_det]
        
        output.append(x[i])
        keepi.append(xk_filtered[i:i].reshape(-1, 1))
        
        if (time.time() - t) > time_limit:
            print(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    return (output, keepi) if return_idxs else output


def numpy_nms(boxes, scores, iou_threshold):
    """Pure NumPy NMS implementation.
    
    Args:
        boxes: array of shape (N, 4) in format [x1, y1, x2, y2]
        scores: array of shape (N,)
        iou_threshold: NMS threshold
        
    Returns:
        indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by score descending
    order = np.argsort(-scores)
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate intersection with all remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Calculate width and height
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Calculate intersection area
        inter = w * h
        
        # Calculate union area
        union = areas[i] + areas[order[1:]] - inter
        
        # Calculate IoU
        iou = inter / union
        
        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    
    shape = im.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  
        r = min(r, 1.0)
 
    ratio = r, r  
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    if auto:  
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  
    elif scaleFill:  
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  
 
    dw /= 2  
    dh /= 2
 
    if shape[::-1] != new_unpad:  
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  
    return im, ratio, (dw, dh)

def data_process_cv2(frame, input_shape):
    im0 = cv2.imread(frame)
    img = letterbox(im0, input_shape, auto=False, stride=32)[0]
    org_data = img.copy()
    img = np.ascontiguousarray(img)
    img = np.asarray(img, dtype=np.uint8)
    img = np.expand_dims(img, 0)
    return img, im0, org_data

# Define xywh2xyxy function for converting bounding box format
def xywh2xyxy(x):
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def post_process_yolo(det, im, im0, gn, save_path, img_name):
    detections = []
    if len(det):
        det[:, :4] = scale_boxes(im.shape[:2], det[:, :4], im0.shape).round()
        colors = Colors()
        for *xyxy, conf, cls in reversed(det):
            print("class:",int(cls), "left:%.0f" % xyxy[0],"top:%.0f" % xyxy[1],"right:%.0f" % xyxy[2],"bottom:%.0f" % xyxy[3], "conf:",'{:.0f}%'.format(float(conf)*100))
            int_coords = [int(tensor.item()) for tensor in xyxy]
            detections.append(int_coords)
            c = int(cls)
            label = names[c]
            res_img = plot_one_box(xyxy, im0, label=f'{label}:{conf:.2f}', color=colors(c, True), line_thickness=4)
            cv2.imwrite(f'{save_path}/{img_name}.jpg',res_img)
            # xywh = (xyxy2xywh(np.array(xyxy,dtype=np.float32).reshape(1, 4)) / gn).reshape(-1).tolist()  # normalized xywh
            # line = (cls, *xywh)  # label format
            # with open(f'{save_path}/{img_name}.txt', 'a') as f:
            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')   
    return detections

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a palette derived from Ultralytics color scheme, converting hex codes to RGB.
        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

def plot_one_box(x, im, color=None, label=None, line_thickness=3, steps=2, orig_shape=None):
    # Ensure image is contiguous
    if not im.flags['C_CONTIGUOUS']:
        im = np.ascontiguousarray(im)
    
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(':')) > 1:
            tf = max(tl - 1, 1)  
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)
    return im

def model_load(model):
    providers = ['AxEngineExecutionProvider']
    session = axe.InferenceSession(model, providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [ x.name for x in session.get_outputs()]
    return session, output_names

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = np.arange(w, dtype=dtype) + grid_cell_offset  # shift x
        sy = np.arange(h, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack((sx, sy), axis=-1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w, 1), stride, dtype=dtype))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), axis=dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), axis=dim)  # xyxy bbox


class DFL:
    """
    NumPy implementation of Distribution Focal Loss (DFL) integral module.
    Original paper: Generalized Focal Loss (IEEE TPAMI 2023)
    """
    
    def __init__(self, c1=16):
        """Initialize with given number of distribution channels"""
        self.c1 = c1
        # 初始化权重矩阵（等效于原conv层的固定权重）
        self.weights = np.arange(c1, dtype=np.float32).reshape(1, c1, 1, 1)
        

    def __call__(self, x):
        """
        前向传播逻辑
        参数:
            x: 输入张量，形状为(batch, channels, anchors)
        返回:
            处理后的张量，形状为(batch, 4, anchors)
        """
        b, c, a = x.shape
        
        # 等效于原view->transpose->softmax操作
        x_reshaped = x.reshape(b, 4, self.c1, a)
        x_transposed = np.transpose(x_reshaped, (0, 2, 1, 3))
        x_softmax = np.exp(x_transposed) / np.sum(np.exp(x_transposed), axis=1, keepdims=True)
        
        # 等效卷积操作(通过张量乘积实现)
        conv_result = np.sum(self.weights * x_softmax, axis=1)
        
        return conv_result.reshape(b, 4, a)
    
class YOLO26Detector:
    def __init__(self, model_path, imgsz=[640,640]):
        self.model_path = model_path
        self.session, self.output_names = model_load(self.model_path)
        self.imgsz = imgsz
        self.stride = [16.,32.]
        self.reg_max = 1
        self.nc = len(names)
        self.nl = len(self.stride)
        self.dfl = DFL(self.reg_max)
        self.max_det = 300

    def postprocess(self, preds: np.ndarray) -> np.ndarray:
        """Post-processes YOLO model predictions using NumPy.

        Args:
            preds (np.ndarray): Raw predictions with shape (batch_size, num_anchors, 4 + nc)

        Returns:
            (np.ndarray): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6)
        """
        boxes = preds[:, :, :4]
        scores = preds[:, :, 4:]
        scores_topk, conf, idx = self.get_topk_index(scores, self.max_det)
        
        # Gather corresponding boxes
        boxes_selected = boxes[np.arange(boxes.shape[0])[:, None], idx[:, :, 0].astype(int)]
        
        return np.concatenate([boxes_selected, scores_topk, conf], axis=-1)

    def get_topk_index(self, scores: np.ndarray, max_det: int) -> tuple:
        """Get top-k indices from scores using NumPy.

        Args:
            scores (np.ndarray): Scores array with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (tuple): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape
        k = max_det
        
        # Get max class score for each anchor: shape (batch_size, anchors)
        max_scores = np.max(scores, axis=2)
        
        # Get top-k indices for each batch
        # Using argsort for each batch separately
        output_scores = np.zeros((batch_size, k, 1), dtype=np.float32)
        output_classes = np.zeros((batch_size, k, 1), dtype=np.float32)
        output_indices = np.zeros((batch_size, k, 1), dtype=np.int32)
        
        for b in range(batch_size):
            # Get topk indices from max_scores
            topk_indices = np.argsort(-max_scores[b])[:k]
            
            # Pad if needed
            if len(topk_indices) < k:
                topk_indices = np.pad(topk_indices, (0, k - len(topk_indices)), mode='constant')
            
            # Get scores for topk indices
            topk_scores_array = scores[b, topk_indices]  # shape (k, nc)
            
            # Get class with max score
            class_indices = np.argmax(topk_scores_array, axis=1)
            topk_values = np.max(topk_scores_array, axis=1)
            
            output_scores[b, :, 0] = topk_values
            output_classes[b, :, 0] = class_indices
            output_indices[b, :, 0] = topk_indices
        
        return output_scores, output_classes, output_indices
    
    def detect_objects(self, image, save_path):
        im, im0, org_data = data_process_cv2(image, self.imgsz)
        img_name = os.path.basename(image).split('.')[0]
        x = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        x = [np.transpose(x[i],(0,3,1,2)) for i in range(self.nl)]    #to nchw
        anchors, strides = (np.transpose(x_arr, (1, 0)) for x_arr in make_anchors(x, self.stride, 0.5))
        box = [x[i][:, :self.reg_max * 4, :] for i in range(self.nl)]
        cls = [x[i][:, self.reg_max * 4:, :] for i in range(self.nl)]
        boxes = np.concatenate([box[i].reshape(1, 4 * self.reg_max, -1) for i in range(self.nl)], axis=-1)
        scores = np.concatenate([cls[i].reshape(1, self.nc, -1) for i in range(self.nl)], axis=-1)
        if self.reg_max > 1:
            dbox = dist2bbox(self.dfl(boxes), np.expand_dims(anchors, axis=0), xywh=False, dim=1) * strides
        else:   # 弃用DFL
            dbox = dist2bbox(boxes, np.expand_dims(anchors, axis=0), xywh=False, dim=1) * strides
        y = np.concatenate((dbox, 1/(1 + np.exp(-scores))), axis=1)
        y = y.transpose([0, 2, 1])
        pred = self.postprocess(y)  # Now returns numpy array directly
        pred = non_max_suppression(
            pred,
            0.3,
            0.45,
            None,
            False,
            max_det=300,
            nc=0,
            end2end=True,
            rotated=False,
            return_idxs=None,
        )
        gn = np.array(org_data.shape)[[1, 0, 1, 0]].astype(np.float32)
        res = post_process_yolo(pred[0], org_data, im0, gn, save_path, img_name)
        return res, im0


if __name__ == '__main__':

    detector = YOLO26Detector(model_path='./cow_ax650_npu3_26.axmodel',imgsz=[640,640])
    img_path = './cow_pic'
    det_path='./cow_yolo26_res'
    os.makedirs(det_path, exist_ok=True)
    imgs = glob.glob(f"{img_path}/*.jpg")
    for idx,img in enumerate(imgs):
        print(f"{idx}/{len(imgs)}: {img}")
        det_result, res_img = detector.detect_objects(img,det_path)