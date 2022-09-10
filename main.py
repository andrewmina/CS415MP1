import cv2
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.




kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])/9

im = cv2.imread("lena.png")

