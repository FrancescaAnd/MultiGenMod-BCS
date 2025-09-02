import os
import argparse
import pandas as pd
import SimpleITK as sitk
import numpy as np
from PIL import Image
from radiomics import featureextractor
import logging
from pathlib import Path

logging.getLogger("radiomics").setLevel(logging.ERROR)

def extract_radiomics(df_roi, output_csv, dataset="cbis", bin_width=25):
    ''' extracts 2D radiomic features from ROI patches and masks '''


    print(f"Starting radiomic features extraction for dataset: {dataset.upper()}")
    settings = {
        'binWidth': bin_width,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'enableCExtensions': True,
        'shape2D': True
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()

    features_list = []

    for i, row in df_roi.iterrows():
        if dataset in ["cbis", "inbreast"]:
            image_path = row["cropped_path"]
            mask_path  = row["mask_path"]
            meta = {
                "patient_id": row.get("patient_id"),
                "image_id": row.get("image_id"),
                "lesion": row.get("lesion"),
                "view": row.get("view"),
                "side": row.get("side"),
                "birads": row.get("birads"),
                "label": row.get("label"),
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Check for NaN or None before calling os.path.exists
        if image_path is None or pd.isna(image_path):
            print(f"Skipping {meta.get('image_id', i)}: image_path is None or NaN")
            continue
        if mask_path is None or pd.isna(mask_path):
            print(f"Skipping {meta.get('image_id', i)}: mask_path is None or NaN")
            continue
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Skipping {meta.get('image_id', i)}: paths do not exist")
            continue

        try:
            mask_arr = np.array(Image.open(mask_path))

            # Read images using SimpleITK
            image_sitk = sitk.ReadImage(image_path)
            mask_bin = (mask_arr > 0).astype(np.uint8)
            mask_sitk = sitk.GetImageFromArray(mask_bin)
            mask_sitk.CopyInformation(image_sitk)

            result = extractor.execute(image_sitk, mask_sitk, label=1)

            # Keep only "original" features
            feat_dict = {k: v for k, v in result.items() if k.startswith("original")}
            feat_dict.update(meta)
            features_list.append(feat_dict)

        except Exception as e:
            print(f"Error extracting {meta['image_id']}: {e}")


    if not features_list:
        print("No features extracted.")
        return

    df_features = pd.DataFrame(features_list)
    print(f"Extraction completed: {df_features.shape[0]} ROIs, {df_features.shape[1]} features")

    df_features.to_csv(output_csv, index=False)
    print(f"Saved csv -> {output_csv}")

    print("Radiomics features extraction finished!")

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2D radiomic features from CBIS-DDSM or INbreast ROIs")
    parser.add_argument("--dataset", type=str, choices=["cbis", "inbreast"], required=True, help="Dataset name")
    args = parser.parse_args()

    configs = {
        "cbis": {
            "input": Path("data/CBIS-DDSM/pkl/two_views_roi.pkl"),
            "output": Path("data/cbis_radiomics.csv"),
            "bin_width": 25,
            "aggregate": True,
        },
        "inbreast": {
            "input": Path("data/INbreast/pkl/two_views_roi.pkl"),
            "output": Path("data/inbreast_radiomics.csv"),
            "bin_width": 25,
            "aggregate": True,
        },
    }

    cfg = configs[args.dataset]

    # --- Load ROI dataframe ---#
    if str(cfg["input"]).endswith(".pkl"):
        df_roi = pd.read_pickle(cfg["input"])
    else:
        raise ValueError("Input must be .pkl or .csv")

    # --- Run extraction ---#
    extract_radiomics(
        df_roi,
        cfg["output"],
        args.dataset,
        cfg["bin_width"],
    )

    print(f"Radiomics extracted for {args.dataset}, saved to {cfg['output']}")
