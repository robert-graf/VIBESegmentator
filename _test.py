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


def main():
    # Ensure working directory = script directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    tasks = [
        (99, "99.nii.gz", False),
        (99, "99.nii.gz", False),  # After deleting dataset model dir
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
            "inphase.nii.gz",
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
    main()
