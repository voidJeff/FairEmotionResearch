"""Dataloading for Imagenet Tiny."""
import os
import glob
import gzip

import imageio
import numpy as np
import torch
from ../util import AffectNetDataset
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import dataset, sampler, dataloader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
import sys
from PIL import Image

NUM_TRAIN_CLASSES = 64
NUM_VAL_CLASSES = 16
NUM_TEST_CLASSES = 20
NUM_SAMPLES_PER_CLASS = 600


def load_image(file_path):
    """Loads and transforms an imagenet tiny image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (1, 224, 224)
    """

    x = Image.open(file_path).convert("RGB")

    std_image = Compose(
            [   
                ToTensor(),
                Resize((224,224)),
                Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )
    x = std_image(x)
    return x

class AffectNetMetaDataset(dataset.Dataset):
    """affectnet dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    def __init__(self, batch_size, data_csv):
        """Inits AffectNetMetaDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()

        # read in data
        self._data = pd.read_csv(data_csv)
        self._task_idx = {}
        for i, race in enumerate(np.unique(self._data.race)):
            self._task_idx[i] == race
        
        self._batch_size = batch_size

        # get weight info
        self.weight_dict = {}
        for idx, race in self._task_idx.items():
            temp_filter = self._data.loc[self._data['race'] == race, :]
            self.weight_dict[idx] = compute_class_weight(class_weight='balanced', classes= np.unique(temp_filter.label), y= np.array(temp_filter.label)) #? should we drop the .cpu() here?

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        images_full, labels_full = [], []

        for class_idx in class_idxs:
            # get a class's examples and sample from them
            sampled_file_paths = np.random.default_rng().choice(
                self._data.loc[self._data['race'] == self._task_idx[class_idx], :].img_path,
                size=self._batch_size,
                replace=False
            )
            images = [load_image(file_path) for file_path in sampled_file_paths]
            label = self._data.loc[self._data.img_path.isin(samples),"label"].tolist()
            # split sampled examples into support and query
            images_full.extend(images)
            labels_full.extend([label])

        # aggregate into tensors
        images_full = torch.stack(images_full)  # shape (7*B, C, H, W)
        labels_full = torch.tensor(labels_full)  # shape (7*B)

        return images_full, labels_full


class AffectnetRaceSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, num_its_per_epoch):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = [0,1,2,3,4,5,6]
        self._num_its_per_epoch = num_its_per_epoch

    def __iter__(self):
        return (
            self._split_idxs for _ in range(self._num_its_per_epoch)
        )

    def __len__(self):
        return self._num_its_per_epoch


def identity(x):
    return x


def get_affectnet_meta_dataloader(
        split,
        batch_size,
        task_batch_size,
        data_csv,
        num_its_per_epoch
):
    """Returns a dataloader.DataLoader for affectnet meta.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    if split == "train":
        return dataloader.DataLoader(
            dataset=AffectNetMetaDataset(batch_size, data_csv),
            batch_size=task_batch_size,
            sampler=AffectnetRaceSampler(num_its_per_epoch),
            num_workers=8,
            collate_fn=identity,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
    elif split == "val":
        dev_dataset = AffectNetDataset('../data/affectnet/val_set', train = False, balance = False)
        return dataloader.DataLoader(dev_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=args.num_workers)