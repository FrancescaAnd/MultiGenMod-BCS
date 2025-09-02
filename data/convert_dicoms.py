import os
import numpy as np
import cv2
from glob import glob
import pydicom
import argparse
from matplotlib import pyplot as plt

def convert(dataset):
    '''Converts .dcm images to PNG while preserving folder structure.'''

    if (dataset == "cbis"): 
        PATH = "data/CBIS-DDSM"
        DICOM_DATA_PATH = f"{PATH}/CBIS-DDSM"  
        dicom_files = glob(f"{DICOM_DATA_PATH}/***/**/*.dcm", recursive=True)

    else: 
        PATH = "data/INbreast"
        DICOM_DATA_PATH = f"{PATH}/AllDICOMs/"  
        dicom_files = glob(f"{DICOM_DATA_PATH}/*.dcm", recursive=True)

        
    CONVERTED_DATA_PATH = f"{PATH}/AllPNGs" 
    os.makedirs(CONVERTED_DATA_PATH, exist_ok=True)

    print(f"Images in .dcm format: {len(dicom_files)}")

    for dicom_path in dicom_files:
        try:
            # Try reading the DICOM file
            dicom = pydicom.dcmread(dicom_path, force=True)
            img_array = dicom.pixel_array 

            # Normalize to 8-bit
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0
            img_array = img_array.astype(np.uint8)

            if dataset == "cbis":
                # For CBIS-DDSM, we need to preserve the folder structure
                rel_path = os.path.relpath(dicom_path, DICOM_DATA_PATH)
                rel_dir = os.path.dirname(rel_path)

                out_dir = os.path.join(CONVERTED_DATA_PATH, rel_dir)
                os.makedirs(out_dir, exist_ok=True)

                out_filename = os.path.splitext(os.path.basename(dicom_path))[0] + ".png"
                output_path = os.path.join(out_dir, out_filename)
            else: 
                filename = os.path.splitext(os.path.basename(dicom_path))[0]
                output_path = os.path.join(CONVERTED_DATA_PATH, f"{filename}.png")

            cv2.imwrite(output_path, img_array)

        except Exception as e:
            print(f"Skipping {dicom_path}: {e}")
            continue

    print("Conversion of DICOM to PNG complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM images to PNGs")
    parser.add_argument("--dataset", type=str, choices=["cbis", "inbreast"], required=True)
    args = parser.parse_args()

    print("Conversion of dicom images in progress, please wait...")
    convert(args.dataset)

