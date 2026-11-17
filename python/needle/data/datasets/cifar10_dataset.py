import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        if train:
            X_batches = []
            y_batches = []
            for i in range(1, 6):
                with open(f"{base_folder}/data_batch_{i}", 'rb') as fo:
                    dict_cifar = pickle.load(fo, encoding='bytes')
                    X_batches.append(dict_cifar[b'data'])
                    y_batches.extend(dict_cifar[b'labels'])
            X_batches = np.concatenate(X_batches, axis=0)
            y_batches = np.array(y_batches)
        else:
            with open(f"{base_folder}/test_batch", 'rb') as fo:
                dict_cifar = pickle.load(fo, encoding='bytes')
                X_batches = dict_cifar[b'data']
                y_batches = np.array(dict_cifar[b'labels'])
        X_batches = X_batches.astype(np.float32)/255.0
        X_batches = X_batches.reshape(-1, 3, 32, 32)
        if transforms:
            for i, X in enumerate(X_batches):
                for t in transforms:
                    X_batches[i] = t(X_batches[i])
        self.X = X_batches
        self.y = y_batches
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.X[index], self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION
