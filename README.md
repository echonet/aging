# Age Prediction - Inference Guide

## Overview
This repository contains a script for running inference on a dataset for **Age Prediction from Echocardiogram**. The script uses a **pretrained R(2+1)D model** wrapped in a **RegressionModelWrapper** to predict cardiac/biological age from echocardiograms.

## Requirements
- Python 3.x
- PyTorch Lightning
- Torchvision
- cvair (for data handling and model wrappers)

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
| `DEATH_DT`   | Date on which the patient dies |
| `Age`        | DEATH_DT - StudyDate (in years), this is the label |


### **Important Notes**
- The model is **not trained for Doppler**;
- Only **video echocardiograms** should be used.
- Each **view** (e.g., A4C, PLAX, SC, A2C) has its **own specific pretrained weight file**.
- Inference should be performed **separately for each view**.
- Ensure that only the **included views (A4C, PLAX, SC, A2C)** are present in the manifest file by filtering appropriately.
- The `split` column **must** be set to `test` for inference.
- The `path_column` specifies the exact location of each video.

## Running Inference
The script is fully configurable via command-line arguments, eliminating the need to modify the script directly. Below is an example command to run inference:

### **Example Usage**
```bash
python age_prediction_inference.py \
    --target age \
    --manifest_path "/workspace/justine/echo_age/a4c_no_major_surgeries_in_train_val.csv" \
    --path_column "path_column" \
    --weights_path "/workspace/justine/echo_age/wandb/run-20241226_210015-ayf7wlik/weights/model_best_epoch_val_mae.pt" \
    --save_path "/workspace/justine/echo_age/inference_results.csv" \
    --num_workers 4 \
    --batch_size 64 \
    --gpu_devices 1
```

### **Command Line Arguments**
- `--target`: The output variable to predict (e.g., age).
- `--manifest_path`: Path to the manifest CSV file.
- `--path_column`: Column name in the manifest that contains the file paths.
- `--weights_path`: Path to the pretrained model weights (**Ensure it matches the correct view**).
- `--save_path`: Path where the predictions will be saved.
- `--num_workers`: Number of workers for data loading (default: 4).
- `--batch_size`: Batch size for inference (default: 64).
- `--gpu_devices`: Number of GPUs to use (default: 1).

## Output
After successful inference, the predictions are saved in the specified `--save_path`. The output file will contain the predicted age for each sample in the manifest.

