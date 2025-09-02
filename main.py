import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified entry point for breast cancer pipeline")
    parser.add_argument("--dataset", type=str, choices=["cbis", "inbreast"], required=True,
                        help="Dataset name")
    parser.add_argument("--task", type=str, choices=["convert", "preprocess", "radiomics", "train", "test"],
                        required=True, help="Which step of the pipeline to run")
    args = parser.parse_args()

    task_to_script = {
        "convert": "data/convert_dicoms.py",
        "radiomics": "utils/extract_radiomics.py",
        "train": "train/train.py",
        "test": "eval/test.py",  # <-- add your test script path here
    }

    # Select script depending on task
    if args.task == "preprocess":
        if args.dataset == "cbis":
            script = "data/preprocessing_cbis.py"
        elif args.dataset == "inbreast":
            script = "data/preprocessing_inbreast.py"
        cmd = [sys.executable, script]  # preprocessing doesn’t need --dataset
        print(f"Preprocessing {args.dataset.upper()} dataset...")

    elif args.task == "test":
        script = task_to_script[args.task]

        # Assign dataset-specific paths
        if args.dataset == "cbis":
            ckpt_path = "checkpoints/train/cbis/best_model.pt"
            test_roi_path = Path("data/CBIS-DDSM/pkl/two_views_roi.pkl")
            radiomics_csv = Path("data/cbis_radiomics.csv")
        elif args.dataset == "inbreast":
            ckpt_path = "checkpoints/train/inbreast/best_model.pt"
            test_roi_path = Path("data/INbreast/pkl/two_views_roi.pkl")
            radiomics_csv = Path("data/inbreast_radiomics.csv")
        else:
            raise ValueError(f"No paths defined for dataset {args.dataset}")

        cmd = [sys.executable, script,
               "--dataset", args.dataset,
               "--ckpt_path", ckpt_path,
               "--roi_path", str(test_roi_path),
               "--radiomics_csv", str(radiomics_csv)]
        print(f"Running test evaluation on {args.dataset.upper()} dataset...")

    else:
        script = task_to_script[args.task]
        cmd = [sys.executable, script, "--dataset", args.dataset]
        if args.task == "convert":
            print(f"Converting {args.dataset.upper()} DICOMs to PNGs...")
        elif args.task == "radiomics":
            print(f"Extracting radiomic features from {args.dataset.upper()}...")
        elif args.task == "train":
            print(f"Training model on {args.dataset.upper()} dataset...")

    # Run
    print(f"Running: {script} with dataset {args.dataset.upper()}")
    subprocess.run(cmd, check=True)
