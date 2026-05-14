import onnxruntime as ort
import numpy as np
import cv2
import glob
import os
import argparse
from dataclasses import dataclass

# Class Names
CLASSES = [
    'Drone'
]

@dataclass
class Object:
    bbox: list  # [x0, y0, width, height]
    label: int
    prob: float

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

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

def decode_distributions(feat, reg_max=16):
    prob = softmax(feat, axis=-1)
    dis = np.sum(prob * np.arange(reg_max), axis=-1)
    return dis

def preprocess(image_path, input_size):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image file: {image_path}")
    original_shape = image.shape[:2]
    img = letterbox(image, input_size, auto=False, stride=32)[0]
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img /= 255.0
    return img, original_shape, image

def postprocess(outputs, original_shape, input_size, confidence_threshold, nms_threshold, num_classes, reg_max=16):
    heads = [
        {'output': outputs[0], 'grid_size': input_size[0] // 8, 'stride': 8},
        {'output': outputs[1], 'grid_size': input_size[0] // 16, 'stride': 16},
        {'output': outputs[2], 'grid_size': input_size[0] // 32, 'stride': 32}
    ]
    detections = []
    bbox_channels = 4 * reg_max
    class_channels = num_classes

    for head in heads:
        output = head['output']
        output = np.transpose(output,(0,2,3,1))
        batch_size, grid_h, grid_w, channels = output.shape
        stride = head['stride']
        
        bbox_part = output[:, :, :, :bbox_channels]
        class_part = output[:, :, :, bbox_channels:]
        
        bbox_part = bbox_part.reshape(batch_size, grid_h, grid_w, 4, reg_max)
        bbox_part = bbox_part.reshape(grid_h * grid_w, 4, reg_max)
        class_part = class_part.reshape(batch_size, grid_h * grid_w, class_channels)
        
        for b in range(batch_size):
            for i in range(grid_h * grid_w):
                h = i // grid_w
                w = i % grid_w
                class_scores = class_part[b, i, :]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                box_prob = sigmoid(class_score)
                if box_prob < confidence_threshold:
                    continue
                bbox = bbox_part[i, :, :]
                dis_left = decode_distributions(bbox[0, :], reg_max)
                dis_top = decode_distributions(bbox[1, :], reg_max)
                dis_right = decode_distributions(bbox[2, :], reg_max)
                dis_bottom = decode_distributions(bbox[3, :], reg_max)
                pb_cx = (w + 0.5) * stride
                pb_cy = (h + 0.5) * stride
                x0 = pb_cx - dis_left * stride
                y0 = pb_cy - dis_top * stride
                x1 = pb_cx + dis_right * stride
                y1 = pb_cy + dis_bottom * stride
                scale_x = original_shape[1] / input_size[0]
                scale_y = original_shape[0] / input_size[1]
                x0 = np.clip(x0 * scale_x, 0, original_shape[1] - 1)
                y0 = np.clip(y0 * scale_y, 0, original_shape[0] - 1)
                x1 = np.clip(x1 * scale_x, 0, original_shape[1] - 1)
                y1 = np.clip(y1 * scale_y, 0, original_shape[0] - 1)
                width = x1 - x0
                height = y1 - y0
                detections.append(Object(
                    bbox=[float(x0), float(y0), float(width), float(height)],
                    label=int(class_id),
                    prob=float(box_prob)
                ))

    if len(detections) == 0:
        return []
    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.prob for d in detections])
    class_ids = np.array([d.label for d in detections])

    final_detections = []
    unique_classes = np.unique(class_ids)
    for cls in unique_classes:
        idxs = np.where(class_ids == cls)[0]
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]
        x1_cls = cls_boxes[:, 0]
        y1_cls = cls_boxes[:, 1]
        x2_cls = cls_boxes[:, 0] + cls_boxes[:, 2]
        y2_cls = cls_boxes[:, 1] + cls_boxes[:, 3]
        areas = (x2_cls - x1_cls) * (y2_cls - y1_cls)
        order = cls_scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1_cls[i], x1_cls[order[1:]])
            yy1 = np.maximum(y1_cls[i], y1_cls[order[1:]])
            xx2 = np.minimum(x2_cls[i], x2_cls[order[1:]])
            yy2 = np.minimum(y2_cls[i], y2_cls[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]
        for idx in keep:
            final_detections.append(Object(
                bbox=cls_boxes[idx].tolist(),
                label=int(cls),
                prob=float(cls_scores[idx])
            ))
    return final_detections

def main():
    parser = argparse.ArgumentParser(description="YOLO11 ONNX Inference")
    parser.add_argument('--model', type=str, default='yolo11s_drone.onnx', help='Model path')
    parser.add_argument('--image', type=str, default='43708.jpg', help='Image path')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.45, help='NMS threshold')
    parser.add_argument('--size', type=int, nargs=2, default=[640, 640], help='Input size W H')
    parser.add_argument('--regmax', type=int, default=16, help='DFL reg_max value')
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]    
    img_path = './drone_pic'
    det_path='./drone_yolo11_res'
    os.makedirs(det_path, exist_ok=True)
    imgs = glob.glob(f"{img_path}/*.jpg")
    for idx,img in enumerate(imgs):
        input_tensor, original_shape, original_image = preprocess(img, tuple(args.size))
        outputs = session.run(output_names, {input_name: input_tensor})

        detections = postprocess(
            outputs,
            original_shape,
            tuple(args.size),
            args.conf,
            args.nms,
            len(CLASSES),
            reg_max=args.regmax
        )

        for det in detections:
            bbox = det.bbox
            score = det.prob
            class_id = det.label
            if class_id >= len(CLASSES):
                label = f"cls{class_id}: {score:.2f}"
            else:
                label = f"{CLASSES[class_id]}: {score:.2f}"
            x, y, w, h = map(int, bbox)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(original_image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(f'{det_path}/{os.path.basename(img)}', original_image)
    print(f"结果已保存到 {det_path}")

if __name__ == '__main__':
    main()