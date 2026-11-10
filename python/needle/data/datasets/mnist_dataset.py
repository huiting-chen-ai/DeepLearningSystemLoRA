from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        # Images, header is 16 bytes big endian, same parsing as hw1
        with gzip.open(image_filename, "rb") as image_file_handle:
            image_header_bytes = image_file_handle.read(16)
            magic_number_images, number_of_images, number_of_rows, number_of_cols = struct.unpack(">IIII", image_header_bytes)
            raw_pixel_bytes = image_file_handle.read()

        all_pixels_flat = np.frombuffer(raw_pixel_bytes, dtype=np.uint8)
        # Keep images H x W x C so transforms work on them, C is 1 for grayscale
        images_as_grid = all_pixels_flat.reshape(number_of_images, number_of_rows, number_of_cols, 1)
        # Scale to 0-1
        self.images = images_as_grid.astype(np.float32) / 255.0

        # Labels, header is 8 bytes
        with gzip.open(label_filename, "rb") as label_file_handle:
            label_header_bytes = label_file_handle.read(8)
            magic_number_labels, number_of_labels = struct.unpack(">II", label_header_bytes)
            raw_label_bytes = label_file_handle.read()

        self.labels = np.frombuffer(raw_label_bytes, dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = self.images[index]
        labels = self.labels[index]

        # Single index gives one image, a slice gives a stack, transforms want one at a time
        if images.ndim == 3:
            images = self.apply_transforms(images)
        else:
            images = np.stack([self.apply_transforms(single_image) for single_image in images])

        return images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION
