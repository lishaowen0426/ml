import numpy as np
from kmeans import Kmeans

class ImageQuantizer:

    def __init__(self, b):
        self.b = b # number of bits per pixel

    def quantize(self, img):
        """
        Quantizes an image into 2^b clusters

        Parameters
        ----------
        img : a (H,W,3) numpy array

        Returns
        -------
        quantized_img : a (H,W) numpy array containing cluster indices

        Stores
        ------
        colours : a (2^b, 3) numpy array, each row is a colour

        """

        H, W, _ = img.shape

        pixel = []

        for h in range(H):
            for w in range(W):
                pixel.append(img[h][w])

        pixel = np.array(pixel)

        model = Kmeans(k=2**self.b)

        means = model.fit(pixel)

        labels = model.predict(pixel)

        labels = np.reshape(labels,(H,W))

        self.colours = means

        return labels

    def dequantize(self, quantized_img):
        H, W = quantized_img.shape
        img = np.zeros((H,W,3), dtype='uint8')
        colours = self.colours

        for h in range(H):
            for w in range(W):
                img [h][w] = colours[quantized_img[h][w]]

        return img
