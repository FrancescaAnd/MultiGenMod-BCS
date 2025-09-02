import os
import re
import pandas as pd
from PIL import Image
import glob
import shutil
import cv2
import numpy as np

# Paths
PATH = "data/CBIS-DDSM/"  
PNG_PATH = f"{PATH}AllPNGs/"
CSV_PATH = f"{PATH}metadata.csv"
PREP_PATH = f"{PATH}preprocessed_cbis"
PKL_PATH = f"{PATH}pkl"
os.makedirs(PKL_PATH, exist_ok=True)
os.makedirs(PREP_PATH, exist_ok=True)


# ----------------- Modules-----------#
def birads_to_label(row):
    ''' Create label column: 0 = benign, 1 = malignant '''

    b = row["assessment"]
    if b in [0, 1, 2]:
        return 0  # benign
    elif b == 3:
        return 0  # treat as benign
    elif b == 4:
        return 0 if "BENIGN" in row["pathology"] else 1
    elif b in [5, 6]:
        return 1  # malignant


def get_mask_path(row):
    ''' Given a row from df_roi, return absolute path to the PNG mask '''
    
    image_name = row["image_id"]
    full_img_path = row["full_image_path"]

    if not os.path.exists(full_img_path):
        print(f"[WARN] Full image not found: {full_img_path}")
        return None
    try:
        full_w, full_h = Image.open(full_img_path).size
    except Exception as e:
        return None

    roi_root = os.path.join(PNG_PATH, image_name)
    if not os.path.isdir(roi_root):
        return None

    dated_folders = sorted(
        d for d in os.listdir(roi_root)
        if os.path.isdir(os.path.join(roi_root, d))
    )
    candidate_masks = []
    roi_masks = []  # masks inside ROI mask folders

    for dated in dated_folders:
        dated_path = os.path.join(roi_root, dated)

        roi_mask_dirs = sorted(
            d for d in glob.glob(os.path.join(dated_path, "*ROI mask images*"))
            if os.path.isdir(d)
        )
        if not roi_mask_dirs:
            roi_mask_dirs = sorted(
                d for d in glob.glob(os.path.join(dated_path, "*-ROI mask images*"))
                if os.path.isdir(d)
            )

        for roi_dir in roi_mask_dirs:
            for png_path in sorted(glob.glob(os.path.join(roi_dir, "*.png"))):
                candidate_masks.append(png_path)
                roi_masks.append(png_path)

        cropped_dirs = [
            d for d in os.listdir(dated_path)
            if "cropped images" in d
        ]
        for cropped_dir in cropped_dirs:
            for png_path in sorted(glob.glob(os.path.join(dated_path, cropped_dir, "*.png"))):
                candidate_masks.append(png_path)

    #Try exact size match
    for mask_path in candidate_masks:
        try:
            w, h = Image.open(mask_path).size
            if (w, h) == (full_w, full_h):
                return mask_path
        except Exception as e:
            continue
    #If no exact match, use first ROI mask if exists
    if roi_masks:
        return roi_masks[0]
    # If anything else, first candidate
    if candidate_masks:
        return candidate_masks[0]

    return None

def reorganize_cbis_ddsm(df_roi, output_root, size=(224, 224)):
    ''' Reorganize CBIS-DDSM dataset into a structured folder format.
    and extract patch/mask of the ROIs'''

    os.makedirs(output_root, exist_ok=True)
    df_roi_updated = df_roi.copy()

    for col in ["cropped_path", "mask_path"]:
        if col not in df_roi_updated.columns:
            df_roi_updated[col] = None

    # Group by base mammogram ID
    df_roi_updated["base_id"] = df_roi_updated["image_id"].apply(
        lambda x: "_".join(x.split("_")[:5])
    )

    for base_id, group in df_roi_updated.groupby("base_id"):

        full_path = group["full_image_path"].iloc[0]
        if not os.path.exists(full_path):
            continue

        case_dir = os.path.join(output_root, base_id) # Create folder for this mammogram
        os.makedirs(case_dir, exist_ok=True)

        full_save = os.path.join(case_dir, f"{base_id}.png")
        shutil.copy(full_path, full_save)

        # Update full_image_path for all ROIs in this group
        df_roi_updated.loc[group.index, "full_image_path"] = full_save

        roi_paths = []

        # Process each ROI
        for roi_idx, (idx, roi_row) in enumerate(group.iterrows(), start=1):
            roi_dir = os.path.join(case_dir, f"{roi_idx}")
            os.makedirs(roi_dir, exist_ok=True)
            full_mask_path = roi_row["full_mask_path"]

            mask_full_save = os.path.join(roi_dir, "1-3.png")
            shutil.copy(full_mask_path, mask_full_save)
            df_roi_updated.at[idx, "full_mask_path"] = mask_full_save

            full_img = cv2.imread(full_save, cv2.IMREAD_GRAYSCALE)
            full_mask = cv2.imread(mask_full_save, cv2.IMREAD_GRAYSCALE)

            ys, xs = np.where(full_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                df_roi_updated.at[idx, "cropped_path"] = None
                df_roi_updated.at[idx, "mask_path"] = None
                continue

            # Crop and resize
            cropped_img = cv2.resize(full_img[ys.min():ys.max(), xs.min():xs.max()], size, interpolation=cv2.INTER_AREA)
            cropped_mask = cv2.resize(full_mask[ys.min():ys.max(), xs.min():xs.max()], size, interpolation=cv2.INTER_NEAREST)

            new_img_path = os.path.join(roi_dir, "1-1.png")
            new_mask_path = os.path.join(roi_dir, "1-2.png")
            cv2.imwrite(new_img_path, cropped_img)
            cv2.imwrite(new_mask_path, cropped_mask)

            df_roi_updated.at[idx, "cropped_path"] = new_img_path
            df_roi_updated.at[idx, "mask_path"] = new_mask_path

            roi_paths.append(new_img_path)

        if "roi_paths" not in df_roi_updated.columns:
            df_roi_updated["roi_paths"] = None
        df_roi_updated.at[group.index[0], "roi_paths"] = roi_paths

    return df_roi_updated


def clean_df(df, num_views=2):
    ''' Clean dataframe to ensure each patient has the required number of views.'''

    grouped = df.groupby(["patient_id", "side"])
    for name, group in grouped:
        if len(group) < 2:
            df = df.drop(index=group.index)
       
    if num_views == 4:
        grouped = df.groupby(["patient_id", "view"])
        for name, group in grouped:
            if (len(group) < 2):
                df = df.drop(index=group.index)
   
    df = df.sort_values(by=["patient_id", "side", "view"])
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":

    # -------------- Load PNG filenames --------------- #
    files = sorted([f for f in os.listdir(PNG_PATH) if f.lower().endswith("")])

    # Extract metadata from filenames
    full_names, lesions, splits, patients, sides, views, dates, rois = [], [], [], [], [], [], [], []
    for file in files:
        parts = file.split("_")

        # First element: lesion-split
        lesion_split = parts[0]
        if lesion_split == "Calc-Test":
            lesion = "Calcification"
            split = "Test"
        elif lesion_split == "Calc-Training":
            lesion = "Calcification"
            split = "Training"
        elif lesion_split == "Mass-Training":
            lesion = "Mass"
            split = "Training" 
        elif lesion_split == "Mass-Test":
            lesion = "Mass"
            split = "Test" 

        patient_id = parts[2]  # eg 00038
        side = parts[3]    # LEFT / RIGHT
        view = parts[4]    # CC / MLO
        roi = parts[5] if len(parts) > 5 else None    

        # extract date from the first subfolder 
        folder_path = os.path.join(PNG_PATH, file)
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        date = None
        if subdirs:
            date = subdirs[0].split("-")[0:3]   
            date = "-".join(date)             

        if side.upper() == "LEFT":
            side = "L"
        elif side.upper() == "RIGHT":
            side = "R"
        else:
            side = None

        full_names.append(file)
        lesions.append(lesion)
        splits.append(split)
        patients.append(patient_id)
        sides.append(side)
        views.append(view)
        dates.append(date)
        rois.append(roi)

    df_imgs = pd.DataFrame({
        "image_id": full_names,
        "patient_id": patients,
        "side": sides,
        "view": views,
        "lesion": lesions,
        "split": splits,
        "date": dates,        
        "ROI": rois
    })

    # ---------------- Load CSV: metadata.csv ---------------------- #
    df_csv = pd.read_csv(CSV_PATH, delimiter=',')
    df_csv = df_csv[["Subject ID", "File Location"]] 

    # Adjust the path names
    df_csv['File Location'] = df_csv['File Location'].apply(lambda x: os.path.normpath(x))
    df_csv['File Location'] = df_csv['File Location'].apply(lambda x: x.replace('CBIS-DDSM' + os.sep, '', 1))

    df_csv = df_csv.rename(columns={
        "Subject ID": "image_id",
        "File Location": "image_path"
    })
    df_csv["image_id"] = df_csv["image_id"].astype(str).str.strip()
    df_csv["image_path"] = df_csv["image_path"].apply(lambda x: os.path.join(PNG_PATH+x, "1-1.png"))

    # PNG info with CSV metadata merging
    df_merged = pd.merge(df_imgs, df_csv, on="image_id", how="left")

    # Separate full mammograms and ROIs
    df_full = df_merged[df_merged["ROI"].isna()].copy()  #Full mammography: returns True where the column "ROI" is NaN
    df_roi = df_merged[df_merged["ROI"].notna()].copy()  #ROIs: returns True where the column "ROI" is not NaN


    # ------------------ Process Full Mammographies ------------------ #
    df_roi["mammography_id"] = df_roi["image_id"].apply(lambda x: "_".join(x.split("_")[:5]))
    roi_info = (
        df_roi.groupby("mammography_id")
        .agg(
            num_rois=("image_id", "count"),
            roi_ids=("image_id", list),
        )
        .reset_index()
    )
    df_full = df_full.merge(
        roi_info, left_on="image_id", right_on="mammography_id", how="left"
    ).drop(columns=["mammography_id"])

    # ------------------ Process ROIs ------------------ #
    # Add mammography_path to df_roi
    df_roi = df_roi.merge(
        df_full[["image_id", "image_path"]].rename(
        columns={"image_id": "mammography_id", "image_path": "full_image_path"}),
        on="mammography_id",
        how="left"
    )

    df_roi["cropped_path"] = None   
    df_roi["mask_path"] = None   
    df_roi["full_mask_path"] = df_roi.apply(get_mask_path, axis=1) # Handle the different locations for the masks
    df_roi.drop(columns=["image_path"], inplace=True)

    #-------------- Load calcification and mass CSVs -------------#
    csv_files = [
        ("calcification", f"{PATH}/calc_case_description_train_set.csv"),
        ("calcification", f"{PATH}/calc_case_description_test_set.csv"),
        ("mass", f"{PATH}/mass_case_description_train_set.csv"),
        ("mass", f"{PATH}/mass_case_description_test_set.csv"),
    ]
    all_dfs = []
    for lesion_type, filename in csv_files:
        if not os.path.exists(filename):
            print(f"Skipping {filename} (not found)")
            continue
        df_lesions = pd.read_csv(filename)
        all_dfs.append(df_lesions)

    if not all_dfs:
        raise FileNotFoundError("No CBIS-DDSM CSV files found.")
    df_lesions = pd.concat(all_dfs, ignore_index=True)

    # Select only needed columns
    df_lesions = df_lesions[["patient_id", "image view", "assessment", "pathology", "left or right breast", "cropped image file path"]]
    df_lesions["label"] = df_lesions.apply(birads_to_label, axis=1) # Compute binary birad_label

    df_lesions = df_lesions.rename(columns={
        "image view": "view",
        "assessment": "birads",
    })

    # Extract first part of cropped image file path to match df_roi
    df_lesions["image_id"] = df_lesions["cropped image file path"].apply(lambda x: str(x).split("/")[0])

    df_roi = df_roi.merge(
        df_lesions[["image_id", "birads", "label"]],
        on="image_id",
        how="left"
    )

    # ------------------ Reorganize dataset + Crop/mask of ROIs extraction------------------ #
    df_roi = reorganize_cbis_ddsm(df_roi, PREP_PATH, size=(224, 224))

    # ---------------------------- Clean and save dataframe ------------------------------- #
    two_views_roi = clean_df(df_roi, num_views=2)
    two_views_roi.to_pickle(f"{PKL_PATH}/two_views_roi.pkl")

    print(f"Preprocessing complete!")






