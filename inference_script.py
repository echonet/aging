import argparse
import torch
from pytorch_lightning import Trainer
from torchvision.models.video import r2plus1d_18

from utils import CedarsDataLoader, EchoDataset, RegressionModelWrapper

parser = argparse.ArgumentParser(description="Run age prediction inference on a dataset")
parser.add_argument("--target", type=str, required=True, help="The column (here age) to predict")
parser.add_argument("--manifest_path", type=str, required=True, help="Path to the manifest file")
parser.add_argument("--path_column", type=str, required=True, help="Column name containing paths")
parser.add_argument("--weights_path", type=str, required=True, help="Path to the pretrained model weights")
parser.add_argument("--save_path", type=str, required=True, help="Path to where you'd like to save predictions in a csv")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
parser.add_argument("--gpu_devices", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--n_frames", type=int, default=16, help="Number of frames to sample from each video")
args = parser.parse_args()

print("Initializing dataset and DataLoader...")
test_ds = EchoDataset(
    path_column=args.path_column,
    manifest_path=args.manifest_path,
    targets=[args.target],
    split="test",
    n_frames=args.n_frames,
    augmentations=None,  # No augmentations during inference
)

test_dl = CedarsDataLoader(
    test_ds,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    drop_last=False,
    shuffle=False,
)

print("Initializing model and loading weights...")
backbone = r2plus1d_18(num_classes=1)
model = RegressionModelWrapper(
    backbone,
    output_names=[args.target],
)

weights = torch.load(args.weights_path, map_location="cpu")
state_dict = weights.get("state_dict", weights) if isinstance(weights, dict) else weights
print(model.load_state_dict(state_dict, strict=True))

print("Running inference...")
trainer = Trainer(accelerator="gpu", devices=args.gpu_devices)
results = trainer.predict(model, dataloaders=test_dl)

print(f"Saving predictions to {args.save_path}...")
model.collate_and_save_predictions(
    results,
    save_path=args.save_path,
    dataset_manifest=test_ds.manifest,
    fallback_merge_on=args.path_column,
)
print("Inference completed successfully.")