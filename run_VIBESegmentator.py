from __future__ import annotations

import sys
from pathlib import Path

from TPTBox.segmentation import run_vibeseg

sys.path.append(str(Path(__file__).parent))
import argparse
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VibeSeg whole-body segmentation on a NIfTI image.")

    # run_vibeseg arguments
    parser.add_argument(
        "--img",
        "--img",
        type=Path,
        help="Input NIfTI image (.nii.gz).",
    )

    parser.add_argument(
        "--out_seg",
        "--out_path",
        type=Path,
        help="Output segmentation file (.nii.gz).",
    )

    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite an existing output segmentation.",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index used for inference.",
    )

    parser.add_argument(
        "--ddevice",
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help="Compute device.",
    )

    parser.add_argument(
        "--dataset-id",
        "--dataset_id",
        type=int,
        default=100,
        help="nnU-Net dataset/model identifier.",
    )

    parser.add_argument(
        "--padd",
        type=int,
        default=5,
        help="Padding in voxels added before inference.",
    )

    parser.add_argument(
        "--keep-size",
        action="store_true",
        help="Keep model output resolution instead of resampling back to input space.",
    )

    # forwarded run_inference_on_file arguments
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill holes in the segmentation after inference.",
    )

    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop image to foreground before inference.",
    )

    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Maximum number of folds to ensemble. Uses all folds by default.",
    )

    parser.add_argument(
        "--logits",
        action="store_true",
        help="Return raw softmax logits in addition to the segmentation.",
    )

    parser.add_argument(
        "--mode",
        default="nearest",
        help="Interpolation mode used during resampling.",
    )

    parser.add_argument(
        "--step-size",
        type=float,
        default=0.5,
        help="Sliding-window inference step size fraction.",
    )

    # parser.add_argument(
    #    "--memory-base",
    #    type=float,
    #    default=None,
    #    help="Base GPU memory reservation in MB. Uses model default when omitted.",
    # )
    #
    # parser.add_argument(
    #    "--memory-factor",
    #    type=float,
    #    default=None,
    #    help="Additional memory estimate per million voxels.",
    # )

    parser.add_argument(
        "--memory-max",
        type=int,
        default=990000,
        help="Maximum GPU memory in MB that should be used. Default is remaining GPU memory. Most models have a base size of 6000 + image. Smaller number causes more super-patches",
    )

    parser.add_argument(
        "--wait-till-gpu-percent-is-free",
        type=float,
        default=0.1,
        help="Required free GPU fraction before inference starts.",
    )

    parser.add_argument(
        "--fail-on-missing-memory",
        action="store_true",
        help="Raise an error instead of waiting when insufficient GPU memory is available.",
    )

    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress information.",
    )

    parser.add_argument(
        "--model-path", type=Path, default=None, help="Path containing nnUNet_results folder. Uses default path when omitted."
    )

    return parser


if __name__ == "__main__":
    args = vars(get_parser().parse_args())

    img = args.pop("img")
    out_seg = args.pop("out_seg")

    run_vibeseg(img, out_seg, **args)
