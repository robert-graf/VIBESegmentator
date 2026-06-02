import argparse
from pathlib import Path

from TPTBox.segmentation import run_spineps


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SPINEPS spine segmentation pipeline.")

    # ------------------------------------------------------------------
    # run_spineps arguments
    # ------------------------------------------------------------------

    parser.add_argument(
        "file_path",
        type=Path,
        help="Input NIfTI image.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset root directory used for BIDS path resolution.",
    )

    parser.add_argument(
        "--model-semantic",
        default="t2w",
        help="Semantic segmentation model name or path. (t2w,ct, vibe)",
    )

    parser.add_argument(
        "--model-instance",
        default="instance",
        help="Instance segmentation model name or path. (instance, ct_instance)",
    )

    parser.add_argument(
        "--model-labeling",
        default=None,
        help="Labeling model name. Use 'none' to disable vertebra labeling. (t2w_labeling, ct_labeling)",
    )

    parser.add_argument(
        "--derivative-name",
        default="derivative",
        help="Name of the derivatives output folder.",
    )

    parser.add_argument(
        "--override-semantic",
        action="store_true",
        help="Recompute semantic segmentation even if it already exists.",
    )

    parser.add_argument(
        "--override-instance",
        action="store_true",
        help="Recompute instance segmentation even if it already exists.",
    )

    parser.add_argument(
        "--save-debug-data",
        action="store_true",
        help="Save intermediate debug files.",
    )

    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw model outputs.",
    )

    # parser.add_argument(
    #    "--ignore-compatibility-issues",
    #    action="store_true",
    #    help="Ignore model/image compatibility and BIDS checks.",
    # )

    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU inference.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    # ------------------------------------------------------------------
    # process_img_nii kwargs
    # ------------------------------------------------------------------

    parser.add_argument(
        "--save-modelres-mask",
        action="store_true",
        help="Save semantic segmentation in model resolution.",
    )

    parser.add_argument(
        "--save-softmax-logits",
        action="store_true",
        help="Save averaged softmax logits as NPZ.",
    )

    parser.add_argument(
        "--override-postpair",
        action="store_true",
        help="Recompute postprocessing results.",
    )

    parser.add_argument(
        "--override-ctd",
        action="store_true",
        help="Recompute centroid files.",
    )

    parser.add_argument(
        "--proc-pad-size",
        type=int,
        default=4,
        help="Padding size used during preprocessing.",
    )

    parser.add_argument(
        "--proc-normalize-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize image intensities before inference.",
    )

    # Semantic postprocessing
    parser.add_argument(
        "--proc-sem-crop-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop image before semantic segmentation.",
    )

    parser.add_argument(
        "--proc-sem-n4-bias-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply N4 bias field correction.",
    )

    parser.add_argument(
        "--proc-sem-remove-inferior-beyond-canal",
        action="store_true",
        help="Remove structures inferior to the spinal canal.",
    )

    parser.add_argument(
        "--proc-sem-clean-beyond-largest-bounding-box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep largest semantic bounding box.",
    )

    parser.add_argument(
        "--proc-sem-clean-small-cc-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove small connected-component artifacts.",
    )

    # Instance postprocessing
    parser.add_argument(
        "--proc-inst-corpus-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply vertebral corpus cleanup.",
    )

    parser.add_argument(
        "--proc-inst-clean-small-cc-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove small connected-component artifacts from instances.",
    )

    parser.add_argument(
        "--proc-inst-largest-k-cc",
        type=int,
        default=0,
        help="Keep K largest connected components (0 = disabled).",
    )

    parser.add_argument(
        "--proc-inst-detect-and-solve-merged-corpi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt to split merged vertebral bodies.",
    )

    parser.add_argument(
        "--vertebra-instance-labeling-offset",
        type=int,
        default=2,
        help="Label offset applied during vertebra numbering.",
    )

    # Labeling
    parser.add_argument(
        "--proc-lab-force-no-tl-anomaly",
        action="store_true",
        help="Disable thoracolumbar anomaly handling.",
    )

    # Shared postprocessing
    parser.add_argument(
        "--proc-fill-3d-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill 3D holes in masks.",
    )

    parser.add_argument(
        "--proc-assign-missing-cc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign missing connected components.",
    )

    parser.add_argument(
        "--proc-clean-inst-by-sem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean instance mask using semantic segmentation.",
    )

    parser.add_argument(
        "--proc-vertebra-inconsistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Correct vertebral numbering inconsistencies.",
    )

    parser.add_argument(
        "--log-inference-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log runtime statistics.",
    )

    return parser


if __name__ == "__main__":
    args = vars(get_parser().parse_args())

    if args["model_labeling"] == "none":
        args["model_labeling"] = None

    run_spineps(**args, ignore_compatibility_issues=True)
