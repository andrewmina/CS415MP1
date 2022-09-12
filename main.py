import cv2
import numpy as np



def mean(im, kernel):
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    im_out = np.zeros_like(im)
    height = kernel.shape[0]
    width = kernel.shape[1]

    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y + kernel_size, x:x + kernel_size, c]
                new_value = np.sum(im_patch/(height*width))
                im_out[y, x, c] = new_value
    return im_out


def convolution(im, kernel):
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

    print(kernel)





    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y+kernel_size, x:x+kernel_size, c]
                new_value = np.sum(kernel*im_patch)
                im_out[y, x, c]= new_value
    return im_out

# kernel = np.array([[1,2,3,4, 5],
#                    [6,7,8,9,10],
#                    [11,12,13,14,15],
#                    [16,17,18,19,20],
#                    [21,22,23,24,25]])/9

kernel = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])/9

im = cv2.imread("lena.png")
im = im.astype(float)
test = kernel.shape
# im_out = convolution(im, kernel)
im_out = mean(im, kernel)
im_out = im_out.astype(np.uint8)
cv2.imwrite('output_image.png', im_out)
cv2.imshow("Output", im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

