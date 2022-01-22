# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from abc import ABC
from pathlib import Path

from solo.utils.ffcv_pretrain_dataloader import prepare_ffcv_dataloader, prepare_ffcv_transform


class FFCVPretrainABC(ABC):
    """Abstract pretrain class that returns a train_dataloader using FFCV."""

    def train_dataloader(self):
        """Returns a train dataloader using dali. Supports multi-crop and asymmetric augmentations.

        Returns:
            A train dataloader in the form of a FFCV Loader.
        """

        device_id = self.local_rank
        num_devices = self.trainer.world_size

        # data augmentations
        unique_augs = self.extra_args["unique_augs"]
        transform_kwargs = self.extra_args["transform_kwargs"]
        num_crops_per_aug = self.extra_args["num_crops_per_aug"]
        ffcv_fit_mem = self.extra_args["ffcv_fit_mem"]

        num_workers = self.extra_args["num_workers"]
        data_dir = Path(self.extra_args["data_dir"])
        train_ffcv_dataset = self.extra_args["train_dir"]
        train_ffcv_dataset = data_dir / train_ffcv_dataset

        # handle custom data by creating the needed transforms
        dataset = self.extra_args["dataset"]
        if unique_augs > 1:
            transform = [
                prepare_ffcv_transform(dataset, device_id, **kwargs)
                for kwargs in transform_kwargs
                for _ in num_crops_per_aug
            ]
        else:
            transform = [
                prepare_ffcv_transform(dataset, device_id, **transform_kwargs)
                for _ in num_crops_per_aug
            ]

        train_loader = prepare_ffcv_dataloader(
            train_ffcv_dataset,
            pipelines=transform,
            batch_size=self.batch_size,
            num_workers=num_workers,
            distributed=num_devices > 1,
            fit_mem=ffcv_fit_mem,
        )

        return train_loader


# class ClassificationABC(ABC):
#     """Abstract classification class that returns a train_dataloader and val_dataloader using
#     dali."""

#     def train_dataloader(self) -> DALIGenericIterator:
#         device_id = self.local_rank
#         shard_id = self.global_rank
#         num_shards = self.trainer.world_size

#         num_workers = self.extra_args["num_workers"]
#         dali_device = self.extra_args["dali_device"]
#         data_dir = Path(self.extra_args["data_dir"])
#         train_dir = Path(self.extra_args["train_dir"])

#         # handle custom data by creating the needed pipeline
#         dataset = self.extra_args["dataset"]
#         if dataset in ["imagenet100", "imagenet"]:
#             pipeline_class = NormalPipeline
#         elif dataset == "custom":
#             pipeline_class = CustomNormalPipeline
#         else:
#             raise ValueError(dataset, "is not supported, used [imagenet, imagenet100 or custom]")

#         train_pipeline = pipeline_class(
#             data_dir / train_dir,
#             validation=False,
#             batch_size=self.batch_size,
#             device=dali_device,
#             device_id=device_id,
#             shard_id=shard_id,
#             num_shards=num_shards,
#             num_threads=num_workers,
#         )
#         train_loader = Wrapper(
#             train_pipeline,
#             output_map=["x", "label"],
#             reader_name="Reader",
#             last_batch_policy=LastBatchPolicy.DROP,
#             auto_reset=True,
#         )
#         return train_loader

#     def val_dataloader(self) -> DALIGenericIterator:
#         device_id = self.local_rank
#         shard_id = self.global_rank
#         num_shards = self.trainer.world_size

#         num_workers = self.extra_args["num_workers"]
#         dali_device = self.extra_args["dali_device"]
#         data_dir = Path(self.extra_args["data_dir"])
#         val_dir = Path(self.extra_args["val_dir"])

#         # handle custom data by creating the needed pipeline
#         dataset = self.extra_args["dataset"]
#         if dataset in ["imagenet100", "imagenet"]:
#             pipeline_class = NormalPipeline
#         elif dataset == "custom":
#             pipeline_class = CustomNormalPipeline
#         else:
#             raise ValueError(dataset, "is not supported, used [imagenet, imagenet100 or custom]")

#         val_pipeline = pipeline_class(
#             data_dir / val_dir,
#             validation=True,
#             batch_size=self.batch_size,
#             device=dali_device,
#             device_id=device_id,
#             shard_id=shard_id,
#             num_shards=num_shards,
#             num_threads=num_workers,
#         )

#         val_loader = Wrapper(
#             val_pipeline,
#             output_map=["x", "label"],
#             reader_name="Reader",
#             last_batch_policy=LastBatchPolicy.PARTIAL,
#             auto_reset=True,
#         )
#         return val_loader
