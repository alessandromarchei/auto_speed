#!/usr/bin/env python3

import sys
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


import torch
import onnxruntime as ort

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# OpenCV uses BGR
color_map = {
    1: (0, 0, 255),      # red
    2: (0, 255, 255),    # yellow
    3: (255, 255, 0),    # cyan
}


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

class AutoSpeedONNXInfer:
    def __init__(
        self,
        onnx_path: str,
        input_width: int = 1024,
        input_height: int = 512,
        conf_thres: float = 0.6,
        iou_thres: float = 0.45,
    ):
        self.train_size = (input_width, input_height)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        available = ort.get_available_providers()

        print(f"Available ONNX providers: {available}")

        providers = []
        # if "CUDAExecutionProvider" in available:
        #     providers.append("CUDAExecutionProvider")
        #     print("Using CUDAExecutionProvider for ONNX inference.")
        # else:
        providers.append("CPUExecutionProvider")
        print("Using CPUExecutionProvider for ONNX inference.")

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
        )

        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name

        self.output_names = [
            output.name for output in self.session.get_outputs()
        ]

        print(f"ONNX model     : {onnx_path}")
        print(f"Provider       : {self.session.get_providers()[0]}")
        print(f"Input name     : {self.input_name}")
        print(f"Input shape    : {self.input.shape}")
        print(f"Outputs        : {self.output_names}")
        print(f"Input size     : {input_width} x {input_height}")
        print(f"Conf threshold : {conf_thres}")
        print(f"IoU threshold  : {iou_thres}")

    # ---------------------------------------------------------------------

    def resize_letterbox(self, img: Image.Image):
        target_w, target_h = self.train_size

        orig_w, orig_h = img.size

        scale = min(
            target_w / orig_w,
            target_h / orig_h,
        )

        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        img_resized = img.resize(
            (new_w, new_h),
            Image.BILINEAR,
        )

        padded_img = Image.new(
            "RGB",
            (target_w, target_h),
            (114, 114, 114),
        )

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        padded_img.paste(
            img_resized,
            (pad_x, pad_y),
        )

        return padded_img, scale, pad_x, pad_y

    # ---------------------------------------------------------------------

    def image_to_array(self, image: Image.Image):
        img, scale, pad_x, pad_y = self.resize_letterbox(image)

        img_array = np.asarray(
            img,
            dtype=np.float32,
        )

        img_array /= 255.0

        # HWC -> CHW
        img_array = np.transpose(
            img_array,
            (2, 0, 1),
        )

        # CHW -> NCHW
        img_array = np.expand_dims(
            img_array,
            axis=0,
        )

        img_array = np.ascontiguousarray(
            img_array,
            dtype=np.float32,
        )

        return img_array, scale, pad_x, pad_y

    # ---------------------------------------------------------------------

    @staticmethod
    def xywh2xyxy(boxes):
        x = boxes.copy()

        x[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        x[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        x[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        x[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

        return x

    # ---------------------------------------------------------------------

    @staticmethod
    def compute_iou(box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection_w = np.maximum(0.0, x2 - x1)
        intersection_h = np.maximum(0.0, y2 - y1)

        intersection = intersection_w * intersection_h

        box_area = (
            max(0.0, box[2] - box[0])
            * max(0.0, box[3] - box[1])
        )

        boxes_area = (
            np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
            * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        )

        union = box_area + boxes_area - intersection

        return intersection / np.maximum(union, 1e-6)

    # ---------------------------------------------------------------------

    def nms(self, boxes, scores, iou_threshold):
        """
        Pure NumPy NMS.

        This avoids requiring torch/torchvision just to test the ONNX.
        """

        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int64)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]

            keep.append(i)

            if order.size == 1:
                break

            remaining = order[1:]

            iou = self.compute_iou(
                boxes[i],
                boxes[remaining],
            )

            order = remaining[
                iou <= iou_threshold
            ]

        return np.asarray(
            keep,
            dtype=np.int64,
        )

    # ---------------------------------------------------------------------

    def post_process_predictions(self, raw_predictions):
        """
        Expected AutoSpeed output:

            [1, C, N]

        where typically:

            C = 4 + num_classes

        Example:
            [1, 8, 10752]

        -> transpose
            [10752, 8]
        """

        predictions = np.asarray(raw_predictions)

        if predictions.ndim != 3:
            raise RuntimeError(
                f"Unexpected AutoSpeed output shape: "
                f"{predictions.shape}. Expected [B, C, N]."
            )

        predictions = predictions.squeeze(0).T

        if predictions.shape[1] < 5:
            raise RuntimeError(
                f"Unexpected prediction shape after transpose: "
                f"{predictions.shape}"
            )

        boxes = predictions[:, :4]
        class_logits = predictions[:, 4:]

        # Same behavior as your original inference code.
        class_probs = 1.0 / (
            1.0 + np.exp(-class_logits)
        )

        scores = np.max(
            class_probs,
            axis=1,
        )

        class_ids = np.argmax(
            class_probs,
            axis=1,
        )

        mask = scores > self.conf_thres

        if not np.any(mask):
            return []

        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        boxes = self.xywh2xyxy(boxes)

        # Class-aware NMS.
        #
        # Doing a single NMS across all classes can incorrectly suppress
        # overlapping detections belonging to different classes.
        keep_all = []

        for cls in np.unique(class_ids):
            cls_indices = np.where(
                class_ids == cls
            )[0]

            cls_keep = self.nms(
                boxes[cls_indices],
                scores[cls_indices],
                self.iou_thres,
            )

            keep_all.extend(
                cls_indices[cls_keep].tolist()
            )

        keep_all = sorted(
            keep_all,
            key=lambda idx: scores[idx],
            reverse=True,
        )

        results = []

        for idx in keep_all:
            results.append([
                float(boxes[idx, 0]),
                float(boxes[idx, 1]),
                float(boxes[idx, 2]),
                float(boxes[idx, 3]),
                float(scores[idx]),
                int(class_ids[idx]),
            ])

        return results

    # ---------------------------------------------------------------------

    def inference(self, image: Image.Image):
        orig_w, orig_h = image.size

        img_array, scale, pad_x, pad_y = \
            self.image_to_array(image)

        outputs = self.session.run(
            None,
            {
                self.input_name: img_array
            },
        )

        predictions = self.post_process_predictions(
            outputs[0]
        )

        # Convert coordinates from letterboxed network input
        # back to original image coordinates.
        for pred in predictions:
            pred[0] = (pred[0] - pad_x) / scale
            pred[1] = (pred[1] - pad_y) / scale
            pred[2] = (pred[2] - pad_x) / scale
            pred[3] = (pred[3] - pad_y) / scale

            pred[0] = np.clip(
                pred[0],
                0,
                orig_w - 1,
            )

            pred[1] = np.clip(
                pred[1],
                0,
                orig_h - 1,
            )

            pred[2] = np.clip(
                pred[2],
                0,
                orig_w - 1,
            )

            pred[3] = np.clip(
                pred[3],
                0,
                orig_h - 1,
            )

        return predictions


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_visualization(
    prediction,
    image,
    show_confidence=False,
):
    vis = image.copy()

    for pred in prediction:
        x1, y1, x2, y2, conf, cls = pred

        cls = int(cls)

        color = color_map.get(
            cls,
            (255, 255, 255),
        )

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        if show_confidence:
            label = f"{cls}: {conf:.2f}"

            label_y = max(
                y1 - 8,
                20,
            )

            cv2.putText(
                vis,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return vis


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

def find_png_frames(dataset_path: Path):
    """
    Search recursively so both of these work:

        sequence/
            000001.png
            000002.png

    and:

        sequence/
            camera_front/
                000001.png
                000002.png
    """

    frames = list(
        dataset_path.rglob("*.png")
    )

    # Path sorting works correctly for filenames such as:
    # 000001.png, 000002.png, ...
    frames = sorted(
        frames,
        key=lambda p: str(p.relative_to(dataset_path)),
    )

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(
        description=(
            "Run AutoSpeed ONNX inference on an ordered "
            "sequence of PNG frames."
        )
    )

    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=Path,
        help=(
            "Path to directory containing the sequence "
            "of PNG frames."
        ),
    )

    parser.add_argument(
        "-m",
        "--onnx",
        required=True,
        type=Path,
        help="Path to AutoSpeed ONNX model.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output visualization directory. "
            "Default: <dataset>/autospeed_visualization"
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="Confidence threshold. Default: 0.6",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold. Default: 0.45",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Network input width. Default: 1024",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Network input height. Default: 512",
    )

    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Draw class ID and confidence.",
    )

    parser.add_argument(
        "--vis",
        action="store_true",
        help="Also display frames while processing.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------------

    dataset_path = args.dataset.resolve()
    onnx_path = args.onnx.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {dataset_path}"
        )

    if not dataset_path.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: {dataset_path}"
        )

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model does not exist: {onnx_path}"
        )

    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(
            f"Expected an .onnx model, got: {onnx_path}"
        )

    # ---------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------

    if args.output is None:
        output_path = (
            dataset_path / "autospeed_visualization"
        )
    else:
        output_path = args.output.resolve()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------------
    # Find frames
    # ---------------------------------------------------------------------

    frames = find_png_frames(
        dataset_path
    )

    # Avoid processing previously generated visualizations if output
    # directory happens to be inside the dataset directory.
    output_path_resolved = output_path.resolve()

    frames = [
        frame
        for frame in frames
        if output_path_resolved
        not in frame.resolve().parents
    ]

    if len(frames) == 0:
        raise RuntimeError(
            f"No PNG images found under: {dataset_path}"
        )

    print()
    print(f"Dataset        : {dataset_path}")
    print(f"Frames         : {len(frames)}")
    print(f"Output         : {output_path}")
    print()

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------

    model = AutoSpeedONNXInfer(
        onnx_path=str(onnx_path),
        input_width=args.width,
        input_height=args.height,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    print()
    print("Starting inference...")
    print()

    # ---------------------------------------------------------------------
    # Inference loop
    # ---------------------------------------------------------------------

    total_detections = 0

    for frame_idx, frame_path in enumerate(frames):
        frame_bgr = cv2.imread(
            str(frame_path),
            cv2.IMREAD_COLOR,
        )

        if frame_bgr is None:
            print(
                f"[WARNING] Cannot read image: "
                f"{frame_path}"
            )
            continue

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image_pil = Image.fromarray(
            frame_rgb
        )

        prediction = model.inference(
            image_pil
        )

        total_detections += len(
            prediction
        )

        vis = make_visualization(
            prediction=prediction,
            image=frame_bgr,
            show_confidence=args.show_confidence,
        )

        # Keep the relative directory structure of the dataset.
        relative_path = frame_path.relative_to(
            dataset_path
        )

        output_frame_path = (
            output_path / relative_path
        )

        output_frame_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_frame_path),
            vis,
        )

        if not success:
            print(
                f"[WARNING] Failed to save "
                f"{output_frame_path}"
            )

        print(
            f"[{frame_idx + 1:06d}/{len(frames):06d}] "
            f"{relative_path} "
            f"-> detections: {len(prediction)}"
        )

        if args.vis:
            display = vis

            max_width = 1280

            if display.shape[1] > max_width:
                scale = max_width / display.shape[1]

                display = cv2.resize(
                    display,
                    None,
                    fx=scale,
                    fy=scale,
                )

            cv2.imshow(
                "AutoSpeed ONNX",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Stopped by user.")
                break

    cv2.destroyAllWindows()

    print()
    print("Completed")
    print(f"Frames          : {len(frames)}")
    print(f"Total detections: {total_detections}")
    print(f"Visualizations  : {output_path}")


if __name__ == "__main__":
    main()
