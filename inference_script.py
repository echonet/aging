import torch
import argparse
from pytorch_lightning import Trainer
from torchvision.models.video import r2plus1d_18
from cvair.data.datasets import CedarsDataLoader, EchoDataset
from cvair.training.model_wrappers import RegressionModelWrapper

parser = argparse.ArgumentParser(description="Run age prediction inference on a dataset")
parser.add_argument("--target", type=str, required=True, help="The column (here age) to predict")
parser.add_argument("--manifest_path", type=str, required=True, help="Path to the manifest file")
parser.add_argument("--path_column", type=str, required=True, help="Column name containing paths")
parser.add_argument("--weights_path", type=str, required=True, help="Path to the pretrained model weights")
parser.add_argument("--save_path", type=str, required=True, help="Path to where you'd like to save predictions in a csv")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
parser.add_argument("--gpu_devices", type=int, default=1, help="Number of GPUs to use")
args = parser.parse_args()

print("Initializing dataset and DataLoader...")
test_ds = EchoDataset(
    split="test",
    path_column=args.path_column,
    manifest_path=args.manifest_path,
    targets=[args.target],
)

test_dl = CedarsDataLoader(
    test_ds, num_workers=args.num_workers, batch_size=args.batch_size, drop_last=False, shuffle=False
)

print("Initializing model and loading weights...")
backbone = r2plus1d_18(num_classes=1)
model = RegressionModelWrapper(
    backbone,
    output_names=[args.target],
)

weights = torch.load(args.weights_path)
print(model.load_state_dict(weights))

print("Running inference...")
trainer = Trainer(accelerator="gpu", devices=args.gpu_devices)
results = trainer.predict(model, dataloaders=test_dl)

print(f"Saving predictions to {args.save_path}...")
model.collate_and_save_predictions(results, save_path=args.save_path)
print("Inference completed successfully.")
