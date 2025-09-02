import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

PATH = "data/INbreast" 
CSV_PATH = f"{PATH}/INbreast.csv"
PNG_PATH = f"{PATH}/AllPNGs"
PREP_PATH = f"{PATH}/preprocessed_INbreast"
PKL_PATH = f"{PATH}/pkl"
os.makedirs(PKL_PATH, exist_ok=True)
os.makedirs(PREP_PATH, exist_ok=True)


# ----------------------- Modules --------------------------#
def birads_to_label(birads):
    '''Create label column: 0 = benign, 1 = malignant'''
    if isinstance(birads, str):
        birads = int(birads[0])
    if birads in [1, 2, 3]:
        return 0
    else:
        return 1

def get_sibling_value(element, key_name, value_type):
    '''helper to extract values from XML structure'''

    found_key = False
    for child in element:
        if found_key:
            if value_type == 'real':
                return float(child.text) if child.text else None
            elif value_type == 'integer':
                return int(child.text) if child.text else None
            elif value_type == 'string':
                return child.text
            elif value_type == 'array':
                return child
        if child.tag == 'key' and child.text == key_name:
            found_key = True
    return None

def parse_xml_file(file_path):
    '''Parse XML ROI annotations'''

    tree = ET.parse(file_path)
    root = tree.getroot()
    filename = os.path.splitext(os.path.basename(file_path))[0]
    image_data = {}
    images = root.find('.//array')
    if images is None:
        return {}

    for image in images.findall('./dict'):
        num_rois = get_sibling_value(image, 'NumberOfROIs', 'integer')
        rois = []
        rois_array = get_sibling_value(image, 'ROIs', 'array')
        if rois_array is not None:
            for roi in rois_array.findall('./dict'):
                roi_info = {
                    'Area': get_sibling_value(roi, 'Area', 'real'),
                    'Center': get_sibling_value(roi, 'Center', 'string'),
                    'Dev': get_sibling_value(roi, 'Dev', 'real'),
                    'IndexInImage': get_sibling_value(roi, 'IndexInImage', 'integer'),
                    'Max': get_sibling_value(roi, 'Max', 'real'),
                    'Mean': get_sibling_value(roi, 'Mean', 'real'),
                    'Min': get_sibling_value(roi, 'Min', 'real'),
                    'Name': get_sibling_value(roi, 'Name', 'string'),
                    'NumberOfPoints': get_sibling_value(roi, 'NumberOfPoints', 'integer'),
                    'Point_px': [pt.text for pt in get_sibling_value(roi, 'Point_px', 'array').findall('string')
                                 ] if get_sibling_value(roi, 'Point_px', 'array') is not None else [],
                    'Total': get_sibling_value(roi, 'Total', 'real'),
                    'Type': get_sibling_value(roi, 'Type', 'integer'),
                }
                rois.append(roi_info)
        image_data[filename] = {
            'NumberOfROIs': num_rois if num_rois is not None else 0,
            'ROIs': rois
        }
    return image_data

def parse_all_xml_files(xml_folder):
    '''Parse all XML files in a folder'''

    all_annotations = {}
    for file_name in os.listdir(xml_folder):
        if file_name.endswith('.xml'):
            xml_data = parse_xml_file(os.path.join(xml_folder, file_name))
            all_annotations.update(xml_data)
    return all_annotations

def roi_extraction(df):
    '''Patches/masks of ROIs extraction'''

    roi_records = []
    target_size = (224, 224)

    for _, row in df.iterrows():
        img_path = os.path.join(PNG_PATH, row["filename"])  
        image = Image.open(img_path).convert("L")
        
        mamm_id = row["image_id"].split("_")[0]  
        mamm_folder = os.path.join(PREP_PATH, mamm_id)
        os.makedirs(mamm_folder, exist_ok=True)

        full_image_path = os.path.join(mamm_folder, row["filename"]) 
        if not os.path.exists(full_image_path):
            image.save(full_image_path)

        rois = row["ROIs"]
        if not rois:
            continue

        for roi_idx, roi in enumerate(rois):
            points_str = roi.get("Point_px", [])
            if not points_str:
                continue

            points = []
            for pt in points_str:
                pt = pt.strip("()")
                try:
                    x_str, y_str = pt.split(",")
                    points.append((float(x_str), float(y_str)))
                except:
                    continue
            if not points:
                continue

            # Bounding box + minimum size for small calcifications handling
            xs, ys = zip(*points)
            xmin, ymin, xmax, ymax = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            min_size = 64
            if xmax - xmin < min_size:
                pad = (min_size - (xmax - xmin)) // 2
                xmin = max(0, xmin - pad)
                xmax = min(image.width, xmax + pad)
            if ymax - ymin < min_size:
                pad = (min_size - (ymax - ymin)) // 2
                ymin = max(0, ymin - pad)
                ymax = min(image.height, ymax + pad)

            # Crop + resize patch
            cropped_img = image.crop((xmin, ymin, xmax, ymax))
            scale_x = target_size[0] / (xmax - xmin)
            scale_y = target_size[1] / (ymax - ymin)
            resized_img = cropped_img.resize(target_size, Image.BILINEAR)

            rescaled_points = [(x - xmin, y - ymin) for x, y in points]
            resized_points = [(x * scale_x, y * scale_y) for x, y in rescaled_points]

            resized_mask = Image.new("L", target_size, 0)
            draw = ImageDraw.Draw(resized_mask)
            if len(resized_points) > 1:
                draw.polygon(resized_points, outline=255, fill=255)
            else:
                cx, cy = resized_points[0]
                r = 3
                draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=255, fill=255)

            full_mask = Image.new("L", image.size, 0)
            draw_full = ImageDraw.Draw(full_mask)
            if len(points) > 1:
                draw_full.polygon(points, outline=255, fill=255)
            else:
                cx, cy = points[0]
                r = 3
                draw_full.ellipse((cx-r, cy-r, cx+r, cy+r), outline=255, fill=255)

            roi_folder = os.path.join(mamm_folder, str(roi_idx + 1))
            os.makedirs(roi_folder, exist_ok=True)

            cropped_path = os.path.join(roi_folder, "1-1.png")
            mask_path = os.path.join(roi_folder, "1-2.png")
            full_mask_path = os.path.join(roi_folder, "1-3.png")

            resized_img.save(cropped_path)
            resized_mask.save(mask_path)
            full_mask.save(full_mask_path)

            roi_records.append({
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "lesion": roi.get("Name", None),
                "birads": row.get("birads", None),
                "label": row.get("label", None),
                "cropped_path": cropped_path,
                "mask_path": mask_path,
                "full_mask_path": full_mask_path,
                "full_image_path": full_image_path,
                "side": row["side"],
                "view": row["view"]
            })
    return roi_records

def clean_df(df, num_views=2):    
    ''' Cleans dataframe to ensure each patient has required views '''
      
    grouped = df.groupby(["patient_id", "side"])
    for name, group in grouped:
        #remove cases that don't have two views
        if len(group) < 2:
            df = df.drop(index=group.index)
    
    df = df[df.label.isin([0, 1])]
    
    if num_views == 4:
        grouped = df.groupby(["patient_id", "view"])
        for name, group in grouped:
            #remove cases that don't have two sides
            if (len(group) < 2):
                df = df.drop(index=group.index)
    
    df = df.sort_values(by=["patient_id", "side", "view"])
    df = df.reset_index(drop=True)
    return df



if __name__ == "__main__":

    # -------------- Load PNG filenames --------------- #
    print("Loading PNG filenames...")
    png_files = sorted([f for f in os.listdir(PNG_PATH) if f.lower().endswith(".png")])

    # Extract metadata from filenames
    fname, patients, sides, views = [], [], [], []
    for f in png_files:
        parts = f.split("_")
        fname.append(parts[0])      # image_id 
        patients.append(parts[1])   # patient_id
        sides.append(parts[3])      # side: L/R 

    df_imgs = pd.DataFrame({
        "image_id": fname,
        "patient_id": patients,
        "side": sides,
        "filename": png_files
    })

    # ---------------------- Load CSV metadata -------------------------- #
    print("Loading CSV metadata...")
    df_csv = pd.read_csv(CSV_PATH, delimiter=';')

    df_csv = df_csv[["File Name", "Bi-Rads", "Acquisition date", "View"]] # Select only needed columns
    df_csv = df_csv.rename(columns={
        "File Name": "image_id",
        "Bi-Rads": "birads",
        "Acquisition date": "date",
        "View": "view"
    })
    df_csv["image_id"] = df_csv["image_id"].astype(str).str.strip()
    df_csv['label'] = df_csv['birads'].apply(birads_to_label)

    #--------------------- Parse XML ROI annotations------------------------#
    xml_annotations = parse_all_xml_files(f"{PATH}/AllXML")

    # ---------------- Merge PNG filename info and CSV metadata------------ #
    df_merged = pd.merge(df_imgs, df_csv, on="image_id", how="left")
    df_merged = df_merged.sort_values(by=["patient_id", "date", "side", "view"])
    df_merged["NumberOfROIs"] = df_merged["image_id"].map(
        lambda x: xml_annotations.get(x, {}).get("NumberOfROIs", 0)
    )
    df_merged["ROIs"] = df_merged["image_id"].map(
        lambda x: xml_annotations.get(x, {}).get("ROIs", [])
    )
    # ------------------- Processing ROIs ------------------- #
    print("Processing ROIs...")
    #Patches/masks of ROIs extraction
    df_roi = pd.DataFrame(roi_extraction(df_merged))
    
    #---------------------Update df_merged full image paths-------------------#
    full_img_map = df_roi.groupby("image_id")["full_image_path"].first().to_dict() 
    df_merged["full_image_path"] = df_merged["image_id"].map(full_img_map)

    num_rois_map = df_roi.groupby("image_id").size().to_dict() #Add number of ROIs per mammogram
    df_merged["num_rois"] = df_merged["image_id"].map(num_rois_map).fillna(0).astype(int)
    roi_folders_map = df_roi.groupby("image_id")["cropped_path"].apply(list).to_dict()
    df_merged["roi_folders"] = df_merged["image_id"].map(roi_folders_map).apply(lambda x: x if isinstance(x, list) else [])

   
    # ---------------- Create two-views and four-views DataFrames ---------------- #
    two_views_roi = clean_df(df_roi, num_views=2)
    two_views_roi.to_pickle(f"{PKL_PATH}/two_views_roi.pkl")

    print(f"Preprocessing complete!")




