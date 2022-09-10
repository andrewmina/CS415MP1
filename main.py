import cv2
import numpy as np

def convolution(im, kernal):
<<<<<<< HEAD
    # Use a breakpoint in the code line below to debug your script.
=======
    im_height, im_width, im_channels = im.shape
    kernal_size = kernal.shape[0]
    pad_size = int((kernal_size-1)/2)
    im_padded = np.zeros((im_height+pad_size*2, im_width+pad_size*2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    im_out = np.zeros_like(im)
>>>>>>> bb8783843cb493062595c534b865f33f35984b8a




kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])/9



im = cv2.imread("lena.png")
im = cv2.imread("lena.png")
im_out = convolution(im, kernal)
