import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random

class Augmentation:
    ''' Data augmentations: small rotation, horizontal flip.'''
    def __init__(self, max_rotation=5, hflip_prob=0.5):
        self.max_rotation = max_rotation
        self.hflip_prob = hflip_prob

    def __call__(self, image, mask):
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-self.max_rotation, self.max_rotation)
        image = TF.rotate(image, angle, fill=0)
        mask = TF.rotate(mask, angle, fill=0)

        return image, mask

class MammogramFullDataset(Dataset):
    def __init__(self, df_roi, rad_loader, transform=None, augment = None, skip_missing=True, patient_ids=None):
        
        self.samples = []
        self.rad_loader = rad_loader
        self.transform = transform
        self.augment = augment

        if patient_ids is not None:
            df_roi = df_roi[df_roi["patient_id"].isin(patient_ids)]

        # group by patient + side
        grouped = df_roi.groupby(["patient_id", "side"])
        for (patient_id, side), group in grouped:
            cc_rows = group[group["view"] == "CC"]
            mlo_rows = group[group["view"] == "MLO"]
            if len(cc_rows) == 0 or len(mlo_rows) == 0:
                if skip_missing:
                    continue
            self.samples.append({
                "patient_id": patient_id,
                "side": side,
                "cc_rows": cc_rows,
                "mlo_rows": mlo_rows
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        def load_full_and_mask(rows):
            ''' Load full image + combined mask + average radiomics '''

            first_row = rows.iloc[0]
            full_image = read_image(first_row["full_image_path"]).float() / 255.0  # [1,H,W]
            full_mask = torch.zeros_like(full_image)
            rad_vector = torch.zeros_like(self.rad_loader.get_radiomics_vector(first_row["image_id"]))

            for row in rows.itertuples():
                mask = read_image(row.full_mask_path).float() / 255.0
                # resize mask to match full_image
                mask = F.interpolate(mask.unsqueeze(0), size=full_image.shape[1:], mode='nearest').squeeze(0)
                # combine multiple lesions
                full_mask = torch.clamp(full_mask + mask, 0, 1)

                rad_vector += self.rad_loader.get_radiomics_vector(row.image_id)

            rad_vector /= len(rows)

            if self.augment:
                full_image, full_mask = self.augment(full_image, full_mask)

            if self.transform:
                full_image = self.transform(full_image)
                full_mask = self.transform(full_mask)

            return full_image, full_mask, rad_vector

        cc_full, cc_mask, cc_rad = load_full_and_mask(sample["cc_rows"])
        mlo_full, mlo_mask, mlo_rad = load_full_and_mask(sample["mlo_rows"])

        cc_input = torch.cat([cc_full, cc_mask], dim=0)   
        mlo_input = torch.cat([mlo_full, mlo_mask], dim=0) 

        rad_vector = (cc_rad + mlo_rad) / 2

        return {
            "patient_id": sample["patient_id"],
            "side": sample["side"],
            "cc_input": cc_input,
            "mlo_input": mlo_input,
            "radiomics": rad_vector
        }
        
