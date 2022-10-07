import math
## Andrew Mina UIN: 654560004
## MiniProject1
import cv2
import numpy as np

def convolution(im, kernel):
    print(kernel)
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size-1)/2)
    im_padded = np.zeros((im_height+pad_size*2, im_width+pad_size*2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    kernel_height = kernel.shape[0] ## getting the number of rows in the tuple
    kernel_width = kernel.shape[1] ## getting the number of columns in the tuple

    top = kernel_height-1  ## getting the last row in the kernel
    bottom = 0
    height_swaps = int(kernel_height / 2)
    for i in range(height_swaps):  ## swapping the kernel horizontally first
        first = kernel.take(top,0)
        second = kernel.take(bottom,0)
        kernel[top] = second
        kernel[bottom] = first
        bottom+=1
        top-=1

    width_swaps = int(kernel_width/2)
    left = 0
    right = kernel_width-1
    for i in range(width_swaps):  ## swapping horizontally
        kernel[:,[left,right]] = kernel[:,[right,left]]
        left+=1
        right-=1

    # print(kernel)
    # applying kernel to each patch
    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y+kernel_size, x:x+kernel_size, c]
                new_value = np.sum(kernel*im_patch)
                im_out[y, x, c]= new_value
    return im_out


def median(im, kernel):
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    kernel_height = kernel.shape[0]  ## getting the number of rows in the tuple
    kernel_width = kernel.shape[1]

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y + kernel_size, x:x + kernel_size, c]
                patchList = []

                for i in range(kernel_height):
                    for j in range(kernel_width):
                        patchList.append(im_patch[i,j])

                patchList.sort()
                new_value = patchList[int(len(patchList)/2)]
                im_out[y, x, c] = new_value
    return im_out


## Gaussian function that computes the kernel based on size its given
## then calls convolution function and returns image
def gaussian(im, kernel):
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    kernel_height = kernel.shape[0]  ## getting the number of rows in the tuple
    kernel_width = kernel.shape[1]

    kernelX = - int(kernel_height / 2)
    kernelY = - int(kernel_width / 2)
    sigma = 1
    for i in range(kernel_height):
        for j in range(kernel_width):
            print(kernelX,",", kernelY)
            kernel[i, j] = round((1 / (2 * (3.141) * (sigma ** 2))) * (2.718) ** -(
                        ((kernelX ** 2) + (kernelY ** 2)) / (2 * (sigma ** 2))), 3)
            kernelX += 1
        # kernelX += 1
        if kernelX >= int(kernel_height / 2):
            kernelX = - int(kernel_height / 2)
        kernelY += 1

    im_out = convolution(im, kernel)
    print(kernel)
    return im_out


def opencvTest(im, kernel):
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    kernel_height = kernel.shape[0]  ## getting the number of rows in the tuple
    kernel_width = kernel.shape[1]

    kernelX = - int(kernel_height / 2)
    kernelY = - int(kernel_width / 2)
    sigma = 1
    for i in range(kernel_height):
        for j in range(kernel_width):
            print(kernelX, ",", kernelY)
            kernel[i, j] = round((1 / (2 * (3.141) * (sigma ** 2))) * (2.718) ** -(
                    ((kernelX ** 2) + (kernelY ** 2)) / (2 * (sigma ** 2))), 3)
            kernelX += 1
        # kernelX += 1
        if kernelX >= int(kernel_height / 2):
            kernelX = - int(kernel_height / 2)
        kernelY += 1

    im_out = np.zeros_like(im)
    im_out = cv2.filter2D(im, -1, kernel)
    return im_out


# kernel = np.array([[1,2,3,4, 5],
#                    [6,7,8,9,10],
#                    [11,12,13,14,15],
#                    [16,17,18,19,20],
#                    [21,22,23,24,25]])

# kernel = np.array([[1,1,1],
#                   [1,1,1],
#                   [1,1,1]])/9

# kernel = np.array([[1,1,1,1,1],
#                   [1,1,1,1,1],
#                   [1,1,1,1,1],
#                   [1,1,1,1,1],
#                   [1,1,1,1,1]])/25

# kernelb = np.array([[0,0,0,0,0,0,0],
#                   [0,0,0,0,0,0,0],
#                   [0,0,0,0,0,0,0],
#                   [0,0,0,2,0,0,0],
#                   [0,0,0,0,0,0,0],
#                   [0,0,0,0,0,0,0],
#                   [0,0,0,0,0,0,0]])
#
kernel = np.array([[1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1]])/49

# kernelb = np.array([[0,0,0,0,0],
#                   [0,0,0,0,0],
#                   [0,0,2,0,0],
#                   [0,0,0,0,0],
#                   [0,0,0,0,0]])
#
# kernel = np.array([[1,1,1,1,1],
#                   [1,1,1,1,1],
#                   [1,1,1,1,1],
#                   [1,1,1,1,1],
#                   [1,1,1,1,1]])/25

#
# kernel = np.array([[1,1,1,1,1,1,1,1,1],
#                   [1,1,1,1,1,1,1,1,1],
#                   [1,1,1,1,1,1,1,1,1],
#                   [1,1,1,1,1,1,1,1,1],
#                   [1,1,1,1,1,1,1,1,1],
#                   [1,1,1,1,1,1,1,1,1],
#                   [1,1,1,1,1,1,1,1,1],
#                   [1,1,1,1,1,1,1,1,1],
#                   [1,1,1,1,1,1,1,1,1]])/81

# kernel = np.array([[.003,.013,.022,.013,.003],  ## Gaussian 5X5
#                    [.013,.059,.097,.059,.013],
#                    [.022,.097,.159,.097,.022],
#                    [.013,.059,.097,.059,.013],
#                    [.003,.013,.022,.013,.003]])

# kernel = np.array([[.059,.097,.059],  ## Gaussian 3X3
#                    [.097,.159,.097],
#                    [.059,.097,.059]])

# kernela = np.array([[1,1,1],  ## kernel mean
#                   [1,1,1],
#                   [1,1,1]])/9
#
# kernelb = np.array([[0,0,0],
#                   [0,2,0],
#                   [0,0,0]])
# kernel = kernelb - kernela


# print(kernel)






im = cv2.imread("lena.png")
im = im.astype(float)
test = kernel.shape
# im_out = convolution(im, kernel)
im_out = opencvTest(im, kernel)
im_out = im_out.astype(np.uint8)
cv2.imwrite('openCV7x7.png', im_out)
cv2.imshow("Output", im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

