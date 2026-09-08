#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
import yaml

from Models.model_components.auto_speed.auto_speed_network import AutoSpeedNetwork


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a trained AutoSpeed PyTorch checkpoint to ONNX."
    )

    parser.add_argument(
        "--pth",
        type=Path,
        required=True,
        help="Path to the trained PyTorch checkpoint (.pt/.pth).",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the AutoSpeed YAML config used for training.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ONNX path.",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="n",
        help="AutoSpeed model version. Default: n",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of AutoSpeed classes. Default: 4",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Run onnx.checker after export.",
    )

    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file does not exist: {config_path}"
        )

    print(f"Loading config: {config_path}")

    with config_path.open("r", encoding="utf-8", errors="ignore") as f:
        params = yaml.safe_load(f)

    if params is None:
        params = {}

    if not isinstance(params, dict):
        raise ValueError(
            f"Expected YAML config to contain a dictionary, "
            f"got {type(params).__name__}"
        )

    return params


def export_model(
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    version: str,
    num_classes: int,
    check: bool,
):
    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    params = load_config(config_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Recover architecture parameters exactly as done during training
    # ------------------------------------------------------------------

    encoder_name = params.get("encoder_name", None)
    encoder_pretrained = params.get("encoder_pretrained", False)

    print()
    print("AutoSpeed model configuration")
    print("--------------------------------")
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Config:             {config_path}")
    print(f"Output:             {output_path}")
    print(f"Version:            {version}")
    print(f"Num classes:        {num_classes}")
    print(f"Encoder:            {encoder_name}")
    print(f"Encoder pretrained: {encoder_pretrained}")
    print()

    # ------------------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------------------

    print("Loading trained AutoSpeed model...")

    net_builder = AutoSpeedNetwork()

    model = net_builder.load_model(
        version=version,
        num_classes=num_classes,
        checkpoint_path=str(checkpoint_path),
        encoder_name=encoder_name,
        encoder_pretrained=encoder_pretrained,
    )

    # Export should always happen from the normal PyTorch model in
    # inference mode and FP32.
    model = model.float()
    model.eval()

    # Make sure parameters are on CPU.
    #
    # This avoids unnecessary CUDA dependencies during ONNX export and
    # makes the exporter usable also on machines without a GPU.
    model = model.cpu()

    print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Export ONNX
    # ------------------------------------------------------------------

    print()
    print("Exporting ONNX...")

    with torch.inference_mode():
        net_builder.export_onnx(
            model,
            output_path=str(output_path),
        )

    if not output_path.is_file():
        raise RuntimeError(
            f"ONNX export finished but output file was not created: "
            f"{output_path}"
        )

    # ------------------------------------------------------------------
    # Optional ONNX validation
    # ------------------------------------------------------------------

    if check:
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError(
                "--check was requested but the 'onnx' Python package "
                "is not installed.\n"
                "Install it with:\n"
                "    pip install onnx"
            ) from exc

        print("Checking ONNX model...")

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        print("ONNX checker: OK")

    # ------------------------------------------------------------------
    # Final information
    # ------------------------------------------------------------------

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print()
    print("========================================")
    print("AutoSpeed ONNX export completed")
    print("========================================")
    print(f"Output: {output_path}")
    print(f"Size:   {size_mb:.2f} MB")


def main():
    args = parse_args()

    export_model(
        checkpoint_path=args.pth,
        config_path=args.config,
        output_path=args.output,
        version=args.version,
        num_classes=args.num_classes,
        check=args.check,
    )


if __name__ == "__main__":
    main()