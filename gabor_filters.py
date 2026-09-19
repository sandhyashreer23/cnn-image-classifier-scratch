import cv2
import numpy as np
import torch

def generate_gabor_filters(num_orientations=8, num_scales=4, ksize=3):
    """
    Generates a bank of Gabor filters to match your existing 32-filter first layer
    (8 orientations x 4 scales = 32 filters).
    """
    filters = []
    for scale in range(num_scales):
        sigma = 1.0 + scale * 0.5          # controls the size of the Gaussian envelope
        lambd = 3.0 + scale * 1.5          # controls the wavelength of the sinusoid
        for i in range(num_orientations):
            theta = i * np.pi / num_orientations   # rotate the filter
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma=0.5, psi=0
            )
            filters.append(kernel)

    filters = np.array(filters, dtype=np.float32)          # shape: [32, ksize, ksize]
    filters = torch.tensor(filters).unsqueeze(1)            # add channel dim -> [32, 1, ksize, ksize]
    filters = filters.repeat(1, 3, 1, 1)                    # repeat across RGB -> [32, 3, ksize, ksize]
    return filters

if __name__ == "__main__":
    filters = generate_gabor_filters()
    print("Filter bank shape:", filters.shape)  # should be [32, 3, 3, 3]