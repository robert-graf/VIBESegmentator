import json
import os
import re
import zipfile
from pathlib import Path

# ---------------------------------------------------------
# INPUT PATH
# ---------------------------------------------------------
path = "/DATA/NAS/FASTDATA/robert/nnUNet/nnUNet_results/Dataset540"
p = Path(path)
out_folder = Path("/DATA/NAS/FASTDATA/robert/")

# Extract dataset number (digits at end of path)
m = re.search(r"(\d+)$", p.name)
if not m:
    raise ValueError("Could not extract dataset number from path")
dataset_number = m.group(1)

# Trainer results root:
trainer_dir = p / "nnUNetTrainer__nnUNetPlannerResEncL__3d_fullres"
assert trainer_dir.exists()
# Find folds
folds = sorted([f for f in trainer_dir.iterdir() if f.is_dir() and f.name.startswith("fold_")])
n_folds = len(folds)

print(f"Found {n_folds} folds")

# ---------------------------------------------------------
# PREPARE ZIP NAMES
# ---------------------------------------------------------
suffixes = [""] + [f"_{i}" for i in range(1, n_folds)]  # ["", "_1", "_2", ...]
zip_names = [f"{dataset_number}{s}.zip" for s in suffixes]

# other_downloads.json content
other_dl_content = suffixes[1:]  # ["_1", "_2", ...]


# ---------------------------------------------------------
# HELPER: add file to zip using relative paths
# ---------------------------------------------------------
def add_to_zip(ziph, file_path, base_dir):
    print("Add:", file_path.relative_to(base_dir))
    ziph.write(file_path, arcname=file_path.relative_to(base_dir))


# ---------------------------------------------------------
# CREATE THE ZIP FILES
# ---------------------------------------------------------
for i, fold in enumerate(folds):
    zip_name = out_folder / zip_names[i]
    print(f"Creating: {zip_name}")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        # ---------------------------------------------------------
        # FIRST ZIP ONLY: add top-level JSON + other_downloads.json
        # ---------------------------------------------------------
        if i == 0:
            # Add top-level JSON (*.json)
            for json_file in trainer_dir.glob("*.json"):
                add_to_zip(zipf, json_file, trainer_dir.parent)

            # Write other_downloads.json
            other_file = trainer_dir.parent / "other_downloads.json"
            with open(other_file, "w") as f:
                json.dump(other_dl_content, f, indent=4)
            add_to_zip(zipf, other_file, trainer_dir.parent)
            other_file.unlink()  # remove temporary file

        # ---------------------------------------------------------
        # FOLD CONTENT: checkpoint_final.pth and *.txt
        # ---------------------------------------------------------
        checkpoint = fold / "checkpoint_final.pth"
        if checkpoint.exists():
            add_to_zip(zipf, checkpoint, trainer_dir.parent)

        # for txt_file in fold.glob("*.txt"):
        #    add_to_zip(zipf, txt_file, trainer_dir)
        for txt_file in fold.glob("*.json"):
            add_to_zip(zipf, txt_file, trainer_dir.parent)

print("All ZIP files created.")
