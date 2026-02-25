# Age Prediction - Inference Guide

## Overview
This repository contains a script for running inference on a dataset for **Age Prediction from Echocardiogram**. The script uses a **pretrained R(2+1)D model** wrapped in a **RegressionModelWrapper** to predict cardiac/biological age from echocardiograms.

## Requirements
- Python 3.x
- PyTorch Lightning
- Torchvision

## Manifest Structure
To perform inference, the script requires a **manifest file (CSV)** that lists the videos. This file should include:

### **Required Columns**
| Column Name   | Description  |
|--------------|-------------|
| `file_uid`   | Unique identifier for each file |
| `study_uid`  | Study identifier |
| `view`       | Echocardiogram view |
| `MRN`        | Patient medical record number |
| `series_uid` | Series identifier for the scan (optional, can be NaN) |
| `frames`     | Number of frames in the scan |
| `fps`        | Frames per second (optional, can be NaN)|
| `split`      | **Must be set to 'test' for inference** |
| `path_column` | File paths to echocardiographic videos/images |
| `StudyDate`  | Date on which the Echo Study was done |
| `BirthDate`   | Date on which the patient was born |
| `Age`        | StudyDate - BirthDate (in years), this is the label |


### **Important Notes**
- The model was trained on videos with resolution (112,112)
- The model is **not trained for Doppler**
- Only **video echocardiograms** should be used.
- Each **view** (e.g., A4C, PLAX, SC, A2C) has its **own specific pretrained weight file**.
- Inference should be performed **separately for each view**.
- Ensure that only the **included views (A4C, PLAX, SC, A2C)** are present in the manifest file by filtering appropriately.
- The `split` column **must** be set to `test` for inference.
- The `path_column` specifies the exact location of each video.
- a4c: ['A4C', 'A4C_LV', 'A4C_MV', 'A4C_LA', 'A4C_RV']
- plax: ['PLAX_Zoom_out', 'PLAX', 'PLAX_AV_MV', 'PLAX_RV_inflow', 'PLAX_zoomed_AV', 'PLAX_Proximal_Ascending_Aorta', 'PLAX_zoomed_MV', 'PLAX_RV_outflow']
- a2c: ['A2C', 'A2C_LV']
- sc: ['Subcostal_4C', 'Subcostal_IVC', 'Subcostal_Abdominal_Aorta']
  

## Running Inference
The script is fully configurable via command-line arguments, eliminating the need to modify the script directly. Below is an example command to run inference:

### **Example Usage**
```bash
python inference_script.py \
    --target Age \
    --manifest_path "/workspace/a4c_no_major_surgeries_in_train_val.csv" \
    --path_column "path_column" \
    --weights_path "/workspace/justine/echo_age/wandb/run-20241226_210015-ayf7wlik/weights/model_best_epoch_val_mae.pt" \
    --save_path "/workspace/justine/echo_age/inference_results.csv" \
    --batch_size 32 \
    --gpu_devices 1
```

### **Command Line Arguments**
- `--target`: The output variable to predict (e.g., age).
- `--manifest_path`: Path to the manifest CSV file.
- `--path_column`: Column name in the manifest that contains the file paths.
- `--weights_path`: Path to the pretrained model weights (**Ensure it matches the correct view**).
- `--save_path`: Path where the predictions will be saved.
- `--batch_size`: Batch size for inference (default: 32).
- `--gpu_devices`: Number of GPUs to use (default: 1).

## Output
After successful inference, the predictions are saved in the specified `--save_path`. The output file will contain the predicted age for each sample in the manifest.

## Ensemble Regression for Echocardiographic Age Prediction

Once you have individual predictions for all views, you can ensemble them into a single age prediction per study using the provided ensemble regression script.

### Workflow Overview

1. **Data Loading**: Reads CSV files for each view containing study identifiers, MRN, age, and predicted ages.
2. **Preprocessing**:
    - Fills MRN values to a fixed length.
    - Aggregates predictions per study.
    - Renames columns for clarity.
    - Merges data from all views into a single dataframe.
3. **Model Inference**:
    - Uses mean predictions from each view as features.
    - Loads a pre-trained `HistGradientBoostingRegressor` ensemble model.
    - Predicts age for each study.
4. **Evaluation**:
    - Computes metrics: MAE, MSE, RMSE, R², and Pearson correlation.
    - Prints results for model assessment.

### Usage

Run the ensemble script from the command line:

```bash
python ensmebling.py \
    --plax_csv "plax.csv" \
    --a4c_csv "a4c.csv" \
    --a2c_csv "a2c.csv" \
    --sc_csv "sc.csv" \
    --boosting_weights_path "boosting_weights_path.pkl"
```

Replace the file paths with your actual CSV and model file locations.
