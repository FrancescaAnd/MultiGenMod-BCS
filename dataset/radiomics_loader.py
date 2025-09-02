import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd


class RadiomicsLoader:
    '''Loads and scales radiomic features, providing access by ROI ID '''

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.metadata_cols = ["patient_id", "image_id", "lesion", "view", "side", "birads", "label"]
        self.metadata = df[self.metadata_cols]

        df["image_id"] = df["image_id"].astype(str)

        feature_cols = [c for c in df.columns if c not in self.metadata_cols]
        df_features = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        scaler = StandardScaler()
        self.features = torch.tensor(scaler.fit_transform(df_features), dtype=torch.float32)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

        self.roi_id_to_index = {str(roi_id): idx for idx, roi_id in enumerate(df["image_id"].values)}

    def get_radiomics_vector(self, roi_id):
        roi_id = str(roi_id)  
        idx = self.roi_id_to_index.get(roi_id)
        if idx is None:
            raise ValueError(f"ROI ID {roi_id} not found!")
        return self.features[idx]

    def get_metadata(self, roi_id):
        roi_id = str(roi_id)  
        idx = self.roi_id_to_index.get(roi_id)
        if idx is None:
            raise ValueError(f"ROI ID {roi_id} not found!")
        return self.metadata.iloc[idx]

