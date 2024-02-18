import numpy as np
from skimage import io, measure, morphology
from skimage.io import imsave, imread
from matplotlib import pyplot as plt

img = io.imread('map.png', as_gray=True)
# do thresholding
mask = img < 0.7

plt.matshow(mask, cmap='gray')

# ij coords of perimeter
coords = np.nonzero(mask)
coords
