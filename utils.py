from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from pandas import Series
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from typing_extensions import NotRequired, Required, TypedDict, Unpack

def crop_and_scale(
    img: ArrayLike, res: Tuple[int], interpolation=cv2.INTER_CUBIC, zoom: float = 0.0
) -> ArrayLike:
    """Takes an image, a resolution, and a zoom factor as input, returns the
    zoomed/cropped image."""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    # Crop to correct aspect ratio
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]

    # Apply zoom
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    # Resize image
    img = cv2.resize(img, res, interpolation=interpolation)

    return img

def read_video(
    path: Union[str, Path],
    n_frames: int = None,
    sample_period: int = 1,
    out_fps: float = None,  # Output fps
    fps: float = None,  # input fps of video (default to avi metadata)
    frame_interpolation: bool = True,
    start_frame: int = None,
    random_start: bool = False,
    res: Tuple[int] = None,  # (width, height)
    interpolation=cv2.INTER_CUBIC,
    zoom: float = 0,
) -> ArrayLike:
    """
    Reads a .avi video file at a given path.

    Args:
        path: str, Path -- path to a .avi video file
        n_frames: int -- the number of frames to read.
        sample_period: int -- the stride at which the frames are read, e.g. a value of 2 would mean the video would be "double speed".
        random_start: bool -- whether to start reading the video at a random start point or just the beginning.
        res: tuple of ints -- target resolution to crop the output to.
        interpolation -- kind of interpolation to use when resizing the image. Default is bicubic
        zoom: float -- how much to zoom in, with a value of 0.1 representing a 10% zoom-in.
    Returns:
        out: Numpy array -- array of shape (N x H x W x 3) containing the read, cropped, zoomed video of lenght n_frames.
        vid_size: tuple -- a tuple containing (num_frames, height, width) for the original video (before cropping and scaling)
        fps: float -- the playback rate in frames per second that the original video file contained in its metadata.
    """

    assert sample_period > 0, "sample_period must be a positive integer"
    assert not (
        start_frame is not None and random_start
    ), "random_start requires start_frame to be None"

    # Check path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Get video properties
    cap = cv2.VideoCapture(str(path))
    vid_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if out_fps is not None:
        sample_period = 1
        # Figuring out how many frames to read, and at what stride, to achieve the target
        # output FPS if one is given.
        if n_frames is not None:
            out_n_frames = n_frames
            n_frames = int(np.ceil((n_frames - 1) * fps / out_fps + 1))
        else:
            out_n_frames = int(np.floor((vid_size[0] - 1) * out_fps / fps + 1))

    # Setup output array
    if n_frames is None:
        n_frames = vid_size[0] // sample_period
    if n_frames * sample_period > vid_size[0]:
        raise Exception(
            f"{n_frames} frames requested (with sample period {sample_period}) but video length is only {vid_size[0]} frames"
        )
    if res is None:
        out = np.zeros((n_frames, *vid_size[1:], 3), dtype=np.uint8)
    else:
        out = np.zeros((n_frames, res[1], res[0], 3), dtype=np.uint8)

    # Read video, skipping sample_period frames each time
    if start_frame is not None:
        start_frame = min(vid_size[0] - n_frames * sample_period + 1, start_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if random_start:
        start_frame = np.random.randint(vid_size[0] - n_frames * sample_period + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_i in range(n_frames):
        _, frame = cap.read()
        if res is not None:
            frame = crop_and_scale(frame, res, interpolation, zoom)
        out[frame_i] = frame
        for _ in range(sample_period - 1):
            cap.read()
    cap.release()

    # if a particular output fps is desired, either get the closest frames from the input video
    # or interpolate neighboring frames to achieve the fps without frame stutters.
    if out_fps is not None:
        i = np.arange(out_n_frames) * fps / out_fps
        if frame_interpolation:
            out_0 = out[np.floor(i).astype(int)]
            out_1 = out[np.ceil(i).astype(int)]
            t = (i % 1)[:, None, None, None]
            out = (1 - t) * out_0 + t * out_1
        else:
            out = out[np.round(i).astype(int)]

    if n_frames == 1:
        out = np.squeeze(out)
    return out, vid_size, fps


class Handler:
    """Dummy base class for type hinting."""
    def __init__():
        raise NotImplementedError

    def __call__(self, row: Series) -> Any:
        raise NotImplementedError


class NumberHandler(Handler):
    """
    Args:
        column: name or list of names of column(s) in manifest to return as a tensor.
        dtype: the dtype the return tensor is cast to. Defaults to `torch.float32`.
    """

    def __init__(self, column: str | List[str]):
        self.column = column

        if isinstance(self.column, str):
            self.column = [self.column]

    def __call__(self, row: Series) -> Tensor:
        values = row[self.column].values
        values = np.array(values, dtype=np.float32)
        values = torch.tensor(values, dtype=torch.float32)
        return values


class FileHandlerKwargTypes(TypedDict, total=False):
    """A dummy class used to make IDE autocomplete and tooltips work properly."""

    filename_column: Required[str]
    data_path: NotRequired[Path]
    path_column: NotRequired[str]
    verify_existing: NotRequired[bool]


class FileHandler(Handler):
    """
    Base class for file handlers.
    Args:
        path_column: name of column in manifest that contains the full path to each example's corresponding file.
    """

    def __init__(
        self,
        path_column: str = None,
    ):
        self.path_column = path_column
        assert isinstance(
            self.path_column, str
        ), "FileHandlers only takes a single path column name passed as a string"

    def __call__(self, row: Series) -> Tensor:
        raise NotImplementedError


class VideoHandler(FileHandler):
    def __init__(
        self,
        n_frames: int,
        start_frame_column: str,
        random_start: bool = False,
        sample_rate: int | Tuple[int] | float = 2,
        interpolate_frames: bool = False,
        resize_res: Tuple[int] = None,
        zoom: float = 0,
        **kwargs: Unpack[FileHandlerKwargTypes],
    ):
        super().__init__(**kwargs)

        assert not (
            start_frame_column and random_start
        ), "Cannot have both random_start and start_frame_column"

        self.n_frames = n_frames
        self.start_frame_column = start_frame_column
        self.random_start = random_start
        self.sample_rate = sample_rate
        self.interpolate_frames = interpolate_frames
        self.resize_res = resize_res
        self.zoom = zoom

        if isinstance(self.sample_rate, int):  # Simple single int sample period
            print(f"VideoHandler: Returning 1 out of every {self.sample_rate} frames")

        elif isinstance(self.sample_rate, float):  # Fixed fps
            print(
                f"VideoHandler: Interpolating to fixed framerate of {self.sample_rate}"
            )
        else:
            raise ValueError("sample_rate must be an int or float")

    def __call__(self, row: Series) -> Tensor:
        filepath = Path(row[self.path_column])

        start_frame = (
            row.get(self.start_frame_column, None) if self.start_frame_column else None
        )

        if isinstance(self.sample_rate, int):  # Simple single int sample period
            source_fps = None
            target_fps = None
            sample_period = self.sample_rate

        elif isinstance(self.sample_rate, float):  # Fixed framerate
            source_fps = row["fps"]
            target_fps = self.sample_rate
            sample_period = 1

        try:
            vid, _, _ = read_video(
                filepath,
                self.n_frames,
                fps=source_fps,
                sample_period=sample_period,
                out_fps=target_fps,
                frame_interpolation=self.interpolate_frames,
                random_start=self.random_start,
                start_frame=start_frame,
                res=self.resize_res,
                zoom=self.zoom,
            )
        except Exception as e:
            print(f"Error reading video {filepath}: {e}. Returning None.")
            return None

        vid = torch.from_numpy(vid)
        vid = vid / 255
        vid = torch.movedim(vid, -1, 0).to(torch.float32)

        if len(vid.shape) == 3:
            vid = tv_tensors.Image(vid)
        elif len(vid.shape) == 4:
            vid = tv_tensors.Video(vid)
        else:
            raise ValueError("Video shape must have 3 or 4 dimensions, has ", vid.shape)

        return vid


def cvair_collate_fn(batch: List[dict]):
    """Custom collate function that handles None values and separates manifest metadata."""
    batch = [x for x in batch if x is not None]

    non_rows = []
    rows = []
    for x in batch:
        rows.append(x.pop("manifest_row"))
        non_rows.append(x)

    batch = torch.utils.data.dataloader.default_collate(non_rows)
    batch["manifest_slice"] = pd.DataFrame(rows)

    return batch


class DataLoader(DataLoader):
    """
    A subclass of torch.utils.data.DataLoader that uses the cvair_collate_fn function to collate batches.
    For argument documentation, see torch.utils.data.DataLoader.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("collate_fn", None)
        super().__init__(*args, **kwargs, collate_fn=cvair_collate_fn)


class DatasetKwargTypes(TypedDict, total=False):
    """A dummy class used to make IDE autocomplete and tooltips work properly."""

    manifest_path: Required[Path | str]
    targets: NotRequired[str | Iterable[str]]
    split: NotRequired[str | Iterable[str]]
    subsample: NotRequired[int | float]
    augmentations: NotRequired[
        Callable[[dict], dict] | Iterable[Callable[[dict], dict]] | nn.Module
    ]
    apply_augmentations_to: NotRequired[Iterable[str]]
    drop_na: NotRequired[bool]


class Dataset(Dataset):
    """
    Generic parent class for several different kinds of common datasets we use here at  CVAIR.

    Expects to be used in a scenario where you have a big folder full of input examples (videos, ecgs, 3d arrays, images, etc.)
    and a big CSV that contains metadata and labels for those examples, called a 'manifest'.

    Args:
        manifest_path: Path to a CSV or Parquet file containing the names, labels, and/or metadata of your files.
        split: Allows user to select which split of the manifest to use, assuming the presence of a categorical 'split' column.
        subsample: A number indicating how many examples to randomly subsample from the manifest.
        augmentations: Can be a list of augmentation functions or a single nn.Module.
        apply_augmentations_to: A list of strings indicating which batch elements to apply augmentations to.
    """

    def __init__(
        self,
        manifest_path,
        targets=None,
        split=None,
        subsample=None,
        augmentations=None,
        apply_augmentations_to=("inputs",),
        drop_na=False,
    ):

        self.manifest_path = Path(manifest_path)
        self.targets = targets
        self.split = split
        self.subsample = subsample
        self.augmentations = augmentations
        self.apply_augmentations_to = apply_augmentations_to
        self.drop_na = drop_na

        if isinstance(self.augmentations, nn.Module):
            self.augmentations = [self.augmentations]

        self.manifest = self.read_manifest()

        if self.split is not None:
            if isinstance(self.split, str):
                self.split = [self.split]
            self.manifest = self.manifest[self.manifest["split"].isin(self.split)]
        print(f"Manifest loaded. \nSplit: {self.split}\nLength: {len(self.manifest):,}")

        if self.targets is not None:
            self.default_targets_handler = NumberHandler(column=self.targets)

        self.init_handlers()

        if self.subsample is not None:
            if isinstance(self.subsample, int):
                self.manifest = self.manifest.sample(n=self.subsample)
            else:
                self.subsample = float(self.subsample)
                self.manifest = self.manifest.sample(frac=self.subsample)
            print(f"{subsample} examples subsampled.")

        if self.drop_na:
            subset = None if self.drop_na is True else self.drop_na
            old_len = len(self.manifest)
            self.manifest = self.manifest.dropna(subset=subset)
            new_len = len(self.manifest)
            print(
                f"{old_len - new_len} examples contained NaN value(s) in their labels and were dropped."
            )
        elif not self.drop_na:
            print(
                "drop_na is set to False or None, so it's possible for the manifest to contain NaN values."
            )

        if self.augmentations is not None:
            print(
                "Augmentation will be applied to the following batch items:",
                self.apply_augmentations_to,
            )

    def read_manifest(self) -> pd.DataFrame:
        """Reads manifest file from disk."""
        if self.manifest_path.suffix == ".csv":
            manifest = pd.read_csv(self.manifest_path, low_memory=False)
        elif self.manifest_path.suffix == ".parquet":
            manifest = pd.read_parquet(self.manifest_path)
        elif self.manifest_path.is_dir():
            print(
                "Manifest path is a directory. Dataset will iterate over all files in this directory."
            )
            manifest = pd.DataFrame(
                {"filename": [f.name for f in self.manifest_path.glob("*")]}
            )
        else:
            raise Exception(
                "Manifest path must be a CSV file, Parquet file, or a directory containing files."
            )

        return manifest

    def init_handlers(self):
        """Initialize data handlers. Override this in subclasses to set up specific handlers."""
        pass

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:

        row = self.manifest.iloc[index]

        output_dict = self.produce_example(row)
        if output_dict is None:
            return None

        if self.augmentations is not None:
            output_dict = self.augment(output_dict)

        output_dict["manifest_row"] = row

        return output_dict

    def produce_example(self, row: pd.Series) -> Dict[str, Any]:
        """Produce a single example from a manifest row."""
        output_dict = {}

        if self.targets is not None:
            output_dict["targets"] = self.default_targets_handler(row)
        return output_dict

    def augment(self, output_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:

        if isinstance(self.augmentations, Iterable):
            augmented = {
                k: v for k, v in output_dict.items() if k in self.apply_augmentations_to
            }

            for aug in self.augmentations:
                augmented = aug(augmented)

        elif isinstance(self.augmentations, Callable):
            augmented = self.augmentations(output_dict)

        else:
            raise Exception(
                "self.augmentations must be either an Iterable of augmentations or a single custom augmentation function."
            )

        output_dict.update(augmented)

        return output_dict


class EchoDataset(Dataset):
    """
    Dataset for loading echocardiogram videos.

    Args:
        path_column: Column name containing paths to video files.
        view_threshold: Discard rows where 'view_confidence' column <= this value.
        sample_rate: The temporal stride with which to sample frames.
        interpolate_frames: Whether to interpolate frames to a target FPS.
        resize_res: If not None, will resize the video to this (width, height) pixel resolution.
        zoom: If not None, will crop-zoom the video by this proportion.
    Returns:
        A dictionary containing the video tensor, the label(s), and any other metadata columns from the manifest.
    """

    def __init__(
        self,
        # EchoDataset params
        path_column,
        view_threshold: float = None,
        random_start: bool = False,
        start_frame_column: str = None,
        n_frames: int = 1,
        sample_rate: int | Tuple[int] | float = 2,
        interpolate_frames: bool = False,
        resize_res: Tuple[int] = None,
        zoom: float = 0,
        # Dataset params
        **kwargs: Unpack[DatasetKwargTypes],
    ):
        self.path_column = path_column
        self.view_threshold = view_threshold
        self.random_start = random_start
        self.start_frame_column = start_frame_column
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.interpolate_frames = interpolate_frames
        self.resize_res = resize_res
        self.zoom = zoom

        super().__init__(**kwargs)

    def init_handlers(self):
        super().init_handlers()
        self.echo_handler = VideoHandler(
            path_column=self.path_column,
            n_frames=self.n_frames,
            start_frame_column=self.start_frame_column,
            random_start=self.random_start,
            sample_rate=self.sample_rate,
            interpolate_frames=self.interpolate_frames,
            resize_res=self.resize_res,
            zoom=self.zoom,
        )

    def produce_example(self, row: pd.Series) -> Dict[str, Any]:
        output_dict = super().produce_example(row)
        output_dict["inputs"] = self.echo_handler(row)

        if output_dict["inputs"] is None:
            return None
        else:
            return output_dict

class RegressionModelWrapper(pl.LightningModule):
    """
    Wrapper for regression models with MSE loss.

    Configures Mean Squared Error loss and regression-specific metrics.
    This is a simplified version for inference only.

    Args:
        model: The PyTorch model to wrap
        output_names: Names for each output neuron (string or list of strings).
    """

    def __init__(
        self,
        model: nn.Module,
        output_names: str | List[str],
    ):
        super().__init__()
        self.m = model
        self.output_names = output_names
        if isinstance(self.output_names, str):
            self.output_names = [self.output_names]

        self.loss_func = nn.MSELoss()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Performs forward pass through the model."""
        batch["logits"] = self.m(batch["inputs"])
        return batch

    def prepare_batch(self, batch: dict[str, torch.Tensor]):
        """Prepare batch for forward pass."""
        if "labels" in batch and len(batch["targets"].shape) == 1:
            batch["targets"] = batch["targets"][:, None]
        return batch

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for inference."""
        batch = self.prepare_batch(batch)
        batch = self.forward(batch)
        batch["predictions"] = batch.get("predictions", batch["logits"]).cpu().numpy()
        batch.pop("inputs")

        return batch

    def collate_and_save_predictions(
        self,
        predict_results,
        save_path: Path,
        merge_on=("file_uid",),
        include_labels: bool = False,
        dataset_manifest: pd.DataFrame = None,
        fallback_merge_on: str = None,
    ):
        """Collates prediction results and saves them to a CSV or parquet file.

        Args:
            predict_results: List of prediction results from trainer.predict()
            save_path: Path where predictions file should be saved
            merge_on: Column(s) to merge predictions with original manifest on.
            include_labels: Whether to include ground truth labels in output file.
            dataset_manifest: Optional DataFrame to merge predictions with.
            fallback_merge_on: Optional fallback column to use for merging if merge_on columns are not present.
        """
        if isinstance(merge_on, str):
            merge_on = [merge_on]

        # predict_results is a list of batches, and each batch's 'manifest_slice' key
        # is a DataFrame slice of the original manifest.
        rows = [batch["manifest_slice"] for batch in predict_results]
        metadata_df = pd.concat(rows, axis=0).reset_index(drop=True)

        # Determine which columns to use for merging
        if all(col in metadata_df.columns for col in merge_on):
            merge_cols = list(merge_on)
        elif fallback_merge_on is not None and fallback_merge_on in metadata_df.columns:
            merge_cols = [fallback_merge_on]
        else:
            merge_cols = [metadata_df.columns[0]]

        metadata_df_for_merge = metadata_df[merge_cols]

        if self.output_names is not None:
            columns = [f"{class_name}_preds" for class_name in self.output_names]
        else:
            columns = ["preds"]

        outputs_df = pd.DataFrame(
            np.concatenate([batch["predictions"] for batch in predict_results], axis=0),
            columns=columns,
        )

        if include_labels:
            labels_df = pd.DataFrame(
                np.concatenate([batch["labels"] for batch in predict_results], axis=0),
                columns=[f"{l}_labels" for l in self.output_names],
            )
            outputs_df = pd.concat(
                [outputs_df.reset_index(drop=True), labels_df.reset_index(drop=True)],
                axis=1,
            )

        prediction_df = pd.concat(
            [metadata_df_for_merge.reset_index(drop=True), outputs_df.reset_index(drop=True)],
            axis=1,
        )

        # Merge with manifest (either provided or from dataloader)
        if dataset_manifest is not None:
            manifest = dataset_manifest
        elif hasattr(self, 'trainer') and self.trainer is not None and self.trainer.predict_dataloaders is not None:
            manifest = self.trainer.predict_dataloaders.dataset.manifest
        else:
            manifest = None

        if manifest is not None and all(c in manifest.columns for c in merge_cols):
            prediction_df = prediction_df.merge(manifest, on=merge_cols, how="outer")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.is_dir():
            prediction_df.to_csv(
                save_path / f"predictions.csv",
                index=False,
            )
        else:
            if ".csv" in save_path.name:
                prediction_df.to_csv(
                    save_path,
                    index=False,
                )
            if ".parquet" in save_path.name:
                prediction_df.to_parquet(
                    save_path,
                    index=False,
                )
