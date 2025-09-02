# Multi-view Generative Models for Breast Cancer Screening

## Table of Contents
1. [About the project](#1-about-the-project)
2. [How to use](#2-how-to-use)

## 1. About the project
The aim of the following project is to develop and evaluate a **multi‑view generative model for mammography** to synthesize coherent craniocaudal (CC) and mediolateral oblique (MLO) mammogram views, ensuring cross-view consistency, preservation of lesions and potential improvement in data augmentation and downstream classification tasks.

The project explores **multi-view generative models for mammography**, with a focus on:
- Using **2-view (CC/MLO)** data from **INbreast** and **CBIS-DDSM**.  
- Training **GANs** to generate realistic mammograms (trying **diffusion models** for future improvements).  
- Ensuring **cross-view consistency** and exploring **lesion-conditioned generation**.  
- Supporting **data augmentation** and **explainability** in breast cancer screening.

| Stage                          | Model / Approach          | Input                                 | Conditioning / Details                         | Output                                           |
|--------------------------------|---------------------------|--------------------------------------|-----------------------------------------------|--------------------------------------|
| Multi-View Mammogram Generation | UNet Generator + PatchGAN Discriminator | Single-view mammogram (CC or MLO)    | GAN conditioned on radiomic features          | Synthetic additional view(s) (e.g., generate MLO from CC) |
| Radiomics-Conditioned Synthesis | UNet-based Conditional GAN | Mammogram patch + Radiomic vector    | Radiomics injected as additional input channels | Radiomics-consistent mammogram patch           |


## 2. How to use
This project is implemented on a Linux-based operating system (Ubuntu 22.04.5 LTS, 64-bit).\
To run the code, a working Python environment is required.\
The system utilizes an NVIDIA GeForce RTX GPU, optimized for GPU acceleration with CUDA for enhanced performance.

Thus, It is recommended to create a Conda environment for this purpose.
```shell
conda create --name mammogan python=3.9
```
```shell
conda activate mammogan
```
Once you have downloaded this system from GitHub, navigate inside the `MultiGenMod-BCS` directory (customizing path/to/directory/ with respect where you have located the system)
```shell
cd path/to/directory/MultiGenMod-BCS
```

### Requirements
As first step, please install the dependencies needed for running the system.
```shell
pip install -r requirements.txt
```
### Reproducibility instructions

Two options are available:

1. **Rebuild the full pipeline from raw data**  
   - Download the original datasets and apply all preprocessing steps (conversion, preprocessing, roi extraction).  
   - Provides full control and flexibility, but requires access to raw images.  

2. **Use provided `.pkl` and `.csv` files**  
   - Directly reproduce training and evaluation.  
   - Fast and consistent with reported results.  


#### **OPTION 1: Running the full system**

##### Data collection 
The **INBreast dataset** contains **410 full-field digital mammograms (FFDMs)** from **115 cases (90 patients)** in **DICOM format**. Each image is **expert-annotated** with **ROIs** labeled as **masses, calcifications, distortions, or spiculated regions**. 

INbreast dataset is publicly avaiable at this link: https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset .
Once you have downloaded and unzipped the data, you have to place inside `data/INbreast` folder the following folders from the downloaded dataset:
- AllDICOMs folder
- AllXML folder
- INbreast.csv (if not already in the folder)

The directory should have the following structure:
```graphql
    MultiGenMod-BCS
    │── data/                  # Datasets and preprocessing scripts
    │   │── INbreast/               
    │   │   │── AllDICOMs/            # DICOMs folder
    │   │   │   │── 20586908_6c613....dcm
    │   │   │   │── 20586934_6c613....dcm
    │   │   │    ...
    │   │   │── AllXML/               # XML folder
    │   │   │   │── 20586908.dcm
    │   │   │   │── 20586934.dcm
    │   │   │   ...
    │   │   └── INbreast.csv/         # CSV file
    │   │
    │   │ 
     ...

```

The **CBIS-DDSM dataset** contains **10,239 FFDMs** from **6,671 patients** in **DICOM format**. Each image includes **expert-annotated ROIs** labeled as **masses or calcifications** with **lesion properties** (size, shape, density).  

CBIS-DDSM dataset can be downloaded at this link: https://www.cancerimagingarchive.net/collection/cbis-ddsm
Similarly to INbreast dataset, once you have downloaded and unzipped the data, you have to place inside `data/CBIS-DDSM` folder the following folders from the downloaded dataset:
- CBIS-DDSM folder (containing the DICOM images)
- calc_case_description_test_set.csv (if not already in the folder)
- calc_case_description_train_set.csv
- mass_case_description_train_set.csv
- mass_case_description_test_set.csv
- metadata.csv

The directory should have the following structure:
```graphql
    MultiGenMod-BCS
    │── data/                  # Datasets and preprocessing scripts
    │   │── CBIS-DDSM/               
    │   │   │── CBIS-DDSM/            # DICOMs folders
    │   │   │   │── Calc-Test_P_00038_LEFT_CC_1/
    │   │   │   │── Calc-Test_P_00038_LEFT_CC/
    |   |   |   │── Calc-Test_P_00038_LEFT_MLO_1/
    │   │   │    ...
    │   │   │── calc_case_description_train_set.csv      # all CSVs
    │   │   │── calc_case_description_test_set.csv
    │   │   │── mass_case_description_train_set.csv
    │   │   │── mass_case_description_train_set.csv
    │   │   └── metadata.csv
    │   │   
    │   │
    │   │ 
     ...

```
##### Convert images from DICOM to PNG format
For generating the converted version of INbreast (or of CBIS-DDSM) images from DICOM to PNG format and store them into `data/INbreast/AllPNGs` folder (`data/INbreast/AllPNGs` for CBIS-DDSM), run the following command
```shell
    python main.py --dataset inbreast --task convert
```
```shell
    python main.py --dataset cbis --task convert
```
##### Preprocessing the dataset using PNGs, CSV and XML annotations
For performing the preprocessing on **INbreast**, run the following command:
```shell
    python main.py --dataset inbreast --task preprocess
```
After the execution of this command, 
- the PNG preprocessed images are stored in `data/INbreast/preprocessed_inbreast` directory
- the `two_views_roi.pkl` dataframe is saved in ``data/INbreast/pkl` directory


For performing the preprocessing on **CBIS-DDSM**, run the following command:
```shell
    python main.py --dataset cbis --task preprocess
```
After the execution of this command, 
- the PNG preprocessed images are stored in `data/CBIS-DDSM/preprocessed_cbis` directory
- the `two_views_roi.pkl` dataframe is saved in ``data/CBIS-DDSM/pkl` directory

##### Extract radiomic features
For extracting the radiomic features from preprocessed INbreast/CBIS-DDSM images, run the following command
```shell
    python main.py --dataset inbreast --task radiomics
```
```shell
    python main.py --dataset cbis --task radiomics
```
After the execution of this command, the `inbreast_radiomics.csv` (or `cbis_radiomics.csv`) is saved in `data/` directory

##### Train and evaluate the model
As last step, run the following command to 
- **train** the model saving metrics and statistics in `results.csv` file (`checkpoints/train/inbreast/results.csv`)
- create and save the **checkpoints** (`checkpoints/train/inbreast`)
- **evaluate** the model using validation data and saving visual samples for each epoch (`checkpoints/val/inbreast`)
- save **visualizations** of the output images(`generated_images_full/`)
```shell
    python main.py --dataset inbreast --task train
```
```shell
    python main.py --dataset cbis --task train
```
#### **OPTION 2: Train the system**
Using the CSV, dataframes and metadata already in the folder, train the system running the command above.



