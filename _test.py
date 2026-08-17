#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path


def run_command(cmd: list[str]):
    print(f"\n>>> Running: {' '.join(cmd)}")

    # Live streaming of stdout/stderr
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # print output line by line
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode == 0:
        print("SUCCESS")
        return True
    else:
        print(f"ERROR (exit code {process.returncode})")
        return False


def check_and_delete(path: Path):
    if path.exists():
        print(f"✔ Output file created: {path}")
        path.unlink()
        print(f"✔ Deleted: {path}")
    else:
        print(f"✘ Output file NOT found: {path}")
        exit()


def main_swap():
    # Ensure working directory = script directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    out_file = "swap.nii.gz"
    cmd = [
        "python",
        "run_water_fat_swap_detection.py",
        "--img",
        "vibe_test/fat_vibe.nii.gz",
        "--outphase",
        "vibe_test/outphase_vibe.nii.gz",
        "--inphase",
        "vibe_test/inphase_vibe.nii.gz",
        "--out_seg",
        out_file,
    ]

    success = run_command(cmd)
    if success:
        check_and_delete(Path(out_file))
    else:
        print(f"Command failed, skipping delete check for {out_file}")
        exit()


def main_spineps():
    # Ensure working directory = script directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    tasks = [("vibe", "instance")]

    for semantic, instance in tasks:
        from TPTBox.segmentation import get_outpaths_spineps  # noqa: PLC0415

        try:
            out = get_outpaths_spineps("img.nii.gz", None)
        except ModuleNotFoundError:
            from TPTBox import Print_Logger

            Print_Logger().on_fail("spineps not installed")
            continue

        cmd = [
            "python",
            "run_spine_segmentation.py",
            "img.nii.gz",
            "--model-semantic",
            semantic,
            "--model-instance",
            instance,
        ]

        success = run_command(cmd)
        if success:
            check_and_delete(out["out_spine"])
            check_and_delete(out["out_vert"])
            check_and_delete(out["out_ctd"])
        else:
            print(f"Command failed, skipping delete check for {out_file}")
            exit()


def main_ct():
    # Ensure working directory = script directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    tasks = [(12, "12.nii.gz", False)]

    for dataset_id, out_file, delete in tasks:
        out_path = script_dir / out_file

        # Special case: delete nnUNet results folder
        if delete:
            nnunet_dir = script_dir / "nnUNet" / "nnUNet_results" / f"Dataset{dataset_id:03d}"
            if nnunet_dir.exists():
                print(f"\nDeleting model directory: {nnunet_dir}")
                shutil.rmtree(nnunet_dir)
            else:
                print(f"\nModel directory does not exist: {nnunet_dir}")

        cmd = [
            "python",
            "run_VIBESegmentator.py",
            "--dataset_id",
            str(dataset_id),
            "--img",
            "/media/data/robert/datasets/2022_06_21_T1_CT_wopathfx/dataset/rawdata/fxclass0113/sorted/sub-fxclass0113_sequ-Wssag3mmkf4_ct.nii.gz",
            "--out_path",
            out_file,
            "--override",
        ]

        success = run_command(cmd)
        if success:
            check_and_delete(out_path)
        else:
            print(f"Command failed, skipping delete check for {out_file}")
            exit()


def main():
    # Ensure working directory = script directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    tasks = [
        (99, "99.nii.gz", False),
        (99, "99.nii.gz", True),  # After deleting dataset model dir
        (100, "100.nii.gz", False),
        (100, "100.nii.gz", False),
        (520, "520.nii.gz", False),
        (512, "512.nii.gz", False),
        (278, "512.nii.gz", False),
    ]

    for dataset_id, out_file, delete in tasks:
        out_path = script_dir / out_file

        # Special case: delete nnUNet results folder
        if delete:
            nnunet_dir = script_dir / "nnUNet" / "nnUNet_results" / f"Dataset{dataset_id:03d}"
            if nnunet_dir.exists():
                print(f"\nDeleting model directory: {nnunet_dir}")
                shutil.rmtree(nnunet_dir)
            else:
                print(f"\nModel directory does not exist: {nnunet_dir}")

        cmd = [
            "python",
            "run_VIBESegmentator.py",
            "--dataset_id",
            str(dataset_id),
            "--img",
            "vibe_test/inphase_vibe.nii.gz",
            "--out_path",
            out_file,
            "--override",
        ]

        success = run_command(cmd)
        if success:
            check_and_delete(out_path)
        else:
            print(f"Command failed, skipping delete check for {out_file}")
            exit()


if __name__ == "__main__":
    main_swap()
    main_spineps()
    main_ct()
    main()
