import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        # Horizontal flip is just reversing the width axis
        if flip_img:
            return img[:, ::-1, :]
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        height, width, channels = img.shape
        pad = self.padding

        # Drop the image in the middle of a zero canvas padded on every side
        padded_img = np.zeros((height + 2 * pad, width + 2 * pad, channels), dtype = img.dtype)
        padded_img[pad:pad + height, pad:pad + width, :] = img

        # Crop back to the original size, shifted off center by the random amount
        row_start = pad + shift_x
        col_start = pad + shift_y
        return padded_img[row_start:row_start + height, col_start:col_start + width, :]
        ### END YOUR SOLUTION
