import cv2
import numpy as np,sys
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import filters
from scipy import misc
import scipy.misc

#this is the function which shrinks the first image in a gaussian pyramid
#this fucntion first creates a 5x5 gaussian kernal and then convolves it with the input image
#The next step is to reduce the size of this blured image by half and this process was completed 6 times
def shrink(img1):
    gaussian_kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 72, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])
    gaussian_blur = filters.convolve(img1,gaussian_kernel, mode='reflect')
    shrink = gaussian_blur[::2, ::2] 

    gaussian_blur_2nd = filters.convolve(shrink,gaussian_kernel, mode='reflect')
    shrink2 = gaussian_blur_2nd[::2, ::2] 

    gaussian_blur_3rd = filters.convolve(shrink2,gaussian_kernel, mode='reflect')
    shrink3 = gaussian_blur_3rd[::2, ::2] 

    gaussian_blur_4th = filters.convolve(shrink3,gaussian_kernel, mode='reflect')
    shrink4 = gaussian_blur_4th[::2, ::2] 

    gaussian_blur_5th = filters.convolve(shrink4,gaussian_kernel, mode='reflect')
    shrink5 = gaussian_blur_5th[::2, ::2] 

    gaussian_blur_6th = filters.convolve(shrink5,gaussian_kernel, mode='reflect')
    shrink6 = gaussian_blur_6th[::2, ::2] 

    return shrink6

#This function creates the laplacian pyramid by frist creating a 5x5 kernal and convolving it with the input image 
#The next step is to create the laplacian image by subtracting the input image by the gaussian blurred image

def laplacian(img2):
    gaussian_kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 72, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])
    gaussian_blur = filters.convolve(img2,gaussian_kernel, mode='reflect')
    laplacian = img2 - gaussian_blur

    laplacian_blur = filters.convolve(laplacian,gaussian_kernel, mode='reflect')
    laplacian_layer2 = laplacian - laplacian_blur
    lap_shrink = laplacian_layer2[::2, ::2] 

    laplacian_blur2 = filters.convolve(lap_shrink,gaussian_kernel, mode='reflect')
    laplacian_layer3 = lap_shrink - laplacian_blur2
    lap_shrink2 = laplacian_layer3[::2, ::2] 

    laplacian_blur3 = filters.convolve(lap_shrink2,gaussian_kernel, mode='reflect')
    laplacian_layer4 = lap_shrink2 - laplacian_blur3
    lap_shrink3 = laplacian_layer4[::2, ::2] 

    laplacian_blur4 = filters.convolve(lap_shrink3,gaussian_kernel, mode='reflect')
    laplacian_layer5 = lap_shrink3 - laplacian_blur4
    lap_shrink4 = laplacian_layer5[::2, ::2] 

    laplacian_blur5 = filters.convolve(lap_shrink4,gaussian_kernel, mode='reflect')
    laplacian_layer6 = lap_shrink4 - laplacian_blur5
    lap_shrink5 = laplacian_layer6[::2, ::2] 

    laplacian_blur6 = filters.convolve(lap_shrink5,gaussian_kernel, mode='reflect')
    laplacian_layer7 = lap_shrink5 - laplacian_blur6
    lap_shrink6 = laplacian_layer7[::2, ::2] 

    return lap_shrink6 



def reconstruct(combined):
    gaussian_kernel = (1/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 72, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])
    alpha = 30

    combined_expanded = scipy.ndimage.interpolation.zoom(combined, 2)
    combined_blur = filters.convolve(combined_expanded,gaussian_kernel, mode='reflect')    
    combined_sharp = combined_expanded + alpha * (combined_expanded - combined_blur)

    combined_expanded2 = scipy.ndimage.interpolation.zoom(combined_expanded, 2)
    combined_blur2 = filters.convolve(combined_expanded2,gaussian_kernel, mode='reflect')    
    combined_sharp2 = combined_expanded2 + alpha * (combined_expanded2 - combined_blur2)

    combined_expanded3 = scipy.ndimage.interpolation.zoom(combined_expanded2, 2)
    combined_blur3 = filters.convolve(combined_expanded3,gaussian_kernel, mode='reflect')    
    combined_sharp3 = combined_expanded3 + alpha * (combined_expanded3 - combined_blur3)

    combined_expanded4 = scipy.ndimage.interpolation.zoom(combined_expanded3, 2)
    combined_blur4 = filters.convolve(combined_expanded4,gaussian_kernel, mode='reflect')    
    combined_sharp4 = combined_expanded4 + alpha * (combined_expanded4 - combined_blur4)

    combined_expanded5 = scipy.ndimage.interpolation.zoom(combined_expanded4, 2)
    combined_blur5 = filters.convolve(combined_expanded5,gaussian_kernel, mode='reflect')    
    combined_sharp5 = combined_expanded5 + alpha * (combined_expanded5 - combined_blur5)


    return combined_sharp5

'''Blend the two laplacian pyramids by weighting them according to the mask '''

def blend(img1,img2):

    b, g, r    = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2] # For RGB image
    b2, g2, r2    = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2] # For RGB image

    gaussian_kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 72, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])

    gaussian_blur_b = filters.convolve(b,gaussian_kernel, mode='reflect')
    gaussian_blur_g = filters.convolve(g,gaussian_kernel, mode='reflect')
    gaussian_blur_r = filters.convolve(r,gaussian_kernel, mode='reflect')

    gaussian_blur_b2 = filters.convolve(b2,gaussian_kernel, mode='reflect')
    gaussian_blur_g2 = filters.convolve(g2,gaussian_kernel, mode='reflect')
    gaussian_blur_r2 = filters.convolve(r2,gaussian_kernel, mode='reflect')

    laplacian_b = gaussian_blur_b - b
    laplacian_g = gaussian_blur_g - g
    laplacian_r = gaussian_blur_r - r

    laplacian_b2 = gaussian_blur_b2 - b2
    laplacian_g2 = gaussian_blur_g2 - g2
    laplacian_r2 = gaussian_blur_r2 - r2

    mixed_laplacian_b = laplacian_b+laplacian_b2
    mixed_laplacian_g = laplacian_g+laplacian_g2
    mixed_laplacian_r = laplacian_r+laplacian_r2

    mixed_gaussian_b = gaussian_blur_b+gaussian_blur_b2
    mixed_gaussian_g = gaussian_blur_g+gaussian_blur_g2
    mixed_gaussian_r = gaussian_blur_r+gaussian_blur_r2

    blended_img_b = mixed_gaussian_b+mixed_laplacian_b
    blended_img_g = mixed_laplacian_g+mixed_laplacian_g
    blended_img_r = mixed_laplacian_r+mixed_laplacian_r

    rgb_uint8 = (np.dstack((blended_img_b,blended_img_g,blended_img_r))) .astype(np.uint8)   

    return rgb_uint8


img1 = misc.imread('apple.jpg', flatten=1)
img2 = misc.imread('orange.jpg', flatten=1)
img_color_1 = misc.imread('apple.jpg')
img_color_2 = misc.imread('orange.jpg')
mask = misc.imread('mask.png', flatten=1)



gaussain_pyramid = shrink(img1)
gaussain_pyramid_mask = shrink(mask)
# fig, ax = plt.subplots()
# ax.imshow(gaussain_pyramid, cmap='gray')
# plt.show()
laplacian_pyramid_1 = laplacian(img1)
laplacian_pyramid_2 = laplacian(img2)
# fig, ax = plt.subplots()    
# ax.imshow(laplacian_pyramid, cmap='gray')
# plt.show()

# mask_level_n * source_level_n + (1-mask_level_n) * target_level_n
combined = gaussain_pyramid_mask * laplacian_pyramid_1 + (1-gaussain_pyramid_mask) * laplacian_pyramid_2
blended_img = reconstruct(combined)
fig, ax = plt.subplots()    
ax.imshow(blended_img, cmap='gray')
plt.show()

color_blend = blend(img_color_1,img_color_2)
#img = cv2.imread('Pyramid_blending2.jpg',0)
fig, ax = plt.subplots()
    
ax.imshow(color_blend, cmap='gray')
plt.show()








