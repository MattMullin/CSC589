
import numpy as np
from scipy import ndimage
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import scipy
from scipy import misc
from scipy.misc import imread, imsave
import cv2
from PIL import Image
import pylab as plt
from scipy import signal





# Concatenate three (height, width)s into one (height, width, 3).
def concat_channels(r, g, b):
    assert r.ndim == 2 and g.ndim == 2 and b.ndim == 2
    rgb = (r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis])
    return np.concatenate(rgb, axis=-1)

def hybrid_image_color(im1,im2):
	#Creating the Lowpass Image for the 3 color bands of image 1

	b, g, r    = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2] # For RGB image

	gaussian_kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 36, 48, 36, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])
	im1_red_gaussian = filters.convolve(r, gaussian_kernel, mode="reflect")
	im1_green_gaussian = filters.convolve(g, gaussian_kernel, mode="reflect")
	im1_blue_gaussian = filters.convolve(b, gaussian_kernel, mode="reflect")
	# plt.figure()
	# plt.imshow(im1_red_gaussian,cmap=plt.cm.gray)
	# plt.show()

	#Low pass for the Red
	lpr = np.fft.fft2(im1_red_gaussian)
	lprshift = np.fft.fftshift(lpr) ## shift for centering 0.0 (x,y)
	magnitude_spectrum = 30*np.log(np.abs(lprshift))
	rows = np.size(im1_red_gaussian, 0) #taking the size of the image
	cols = np.size(im1_red_gaussian, 1)
	crow, ccol = rows/2, cols/2
	original_red = np.copy(lprshift)
	lprshift[crow-30:crow+30, ccol-30:ccol+30] = 0
	lpr_ishift= np.fft.ifftshift(original_red - lprshift)
	low_pass_red = np.fft.ifft2(lpr_ishift) ## shift for centering 0.0 (x,y)
	low_pass_red = np.abs(low_pass_red)


	#Low pass for the green
	lpg = np.fft.fft2(im1_green_gaussian)
	lpgshift = np.fft.fftshift(lpg) ## shift for centering 0.0 (x,y)
	magnitude_spectrum = 30*np.log(np.abs(lpgshift))
	rows = np.size(im1_green_gaussian, 0) 
	cols = np.size(im1_green_gaussian, 1)
	crow, ccol = rows/2, cols/2
	original_green = np.copy(lpgshift)
	lpgshift[crow-30:crow+30, ccol-30:ccol+30] = 0
	lpg_ishift= np.fft.ifftshift(original_green - lpgshift)
	low_pass_green = np.fft.ifft2(lpg_ishift) 
	low_pass_green = np.abs(low_pass_green)

	#Low pass for the blue
	lpb = np.fft.fft2(im1_blue_gaussian)
	lpbshift = np.fft.fftshift(lpb) ## shift for centering 0.0 (x,y)
	magnitude_spectrum = 30*np.log(np.abs(lpbshift))
	rows = np.size(im1_blue_gaussian, 0) #taking the size of the image
	cols = np.size(im1_blue_gaussian, 1)
	crow, ccol = rows/2, cols/2
	original_blue = np.copy(lpbshift)
	lpbshift[crow-30:crow+30, ccol-30:ccol+30] = 0
	lpb_ishift= np.fft.ifftshift(original_blue - lpbshift)

	low_pass_blue = np.fft.ifft2(lpb_ishift) ## shift for centering 0.0 (x,y)
	low_pass_blue = np.abs(low_pass_blue)


	#Combining the color channels together to form RGB image
	low_pass_image = im1
	low_pass_image[:, :, 0] =  low_pass_blue
	low_pass_image[:, :, 1] =  low_pass_green
	low_pass_image[:, :, 2] =  low_pass_red

	# plt.figure()
	# plt.imshow(low_pass_image)
	# plt.show()


	#creating the High Pass Image for the 3 color bands in Image 2

	b2, g2, r2    = im2[:, :, 0], im2[:, :, 1], im2[:, :, 2] # For RGB image


	# im2_red_gaussian = filters.convolve(r2, gaussian_kernel, mode="reflect")
	# im2_green_gaussian = filters.convolve(g2, gaussian_kernel, mode="reflect")
	# im2_blue_gaussian = filters.convolve(b2, gaussian_kernel, mode="reflect")

	#Creating the High pass filter for the red band
	f_red = np.fft.fft2(r2)
	red_high_shift = np.fft.fftshift(f_red) ## shift for centering 0.0 (x,y)
	magnitude_spectrum = 60*np.log(np.abs(red_high_shift))

	## removing low frequency contents by applying a 60x60 rectangle window (for masking)
	rows = np.size(r2, 0) #taking the size of the image
	cols = np.size(r2, 1)
	crow, ccol = rows/2, cols/2

	red_high_shift[crow-10:crow+10, ccol-10:ccol+10] = 0
	red_high_ishift= np.fft.ifftshift(red_high_shift)

	highpass_red = np.fft.ifft2(red_high_ishift) ## shift for centering 0.0 (x,y)
	highpass_red = np.abs(highpass_red)


	#Creating the High pass filter for the green band
	f_green = np.fft.fft2(g2)
	green_high_shift = np.fft.fftshift(f_green) ## shift for centering 0.0 (x,y)
	magnitude_spectrum = 60*np.log(np.abs(green_high_shift))

	## removing low frequency contents by applying a 60x60 rectangle window (for masking)
	rows = np.size(g2, 0) #taking the size of the image
	cols = np.size(g2, 1)
	crow, ccol = rows/2, cols/2

	green_high_shift[crow-10:crow+10, ccol-10:ccol+10] = 0
	green_high_ishift= np.fft.ifftshift(green_high_shift)

	highpass_green = np.fft.ifft2(green_high_ishift) ## shift for centering 0.0 (x,y)
	highpass_green = np.abs(highpass_green)

	#Creating the High pass filter for the blue band
	f_blue = np.fft.fft2(b2)
	blue_high_shift = np.fft.fftshift(f_blue) ## shift for centering 0.0 (x,y)
	magnitude_spectrum = 60*np.log(np.abs(blue_high_shift))

	## removing low frequency contents by applying a 60x60 rectangle window (for masking)
	rows = np.size(b2, 0) #taking the size of the image
	cols = np.size(b2, 1)
	crow, ccol = rows/2, cols/2

	blue_high_shift[crow-10:crow+10, ccol-10:ccol+10] = 0
	blue_high_ishift= np.fft.ifftshift(blue_high_shift)

	highpass_blue = np.fft.ifft2(blue_high_ishift) ## shift for centering 0.0 (x,y)
	highpass_blue = np.abs(highpass_blue)

	#Combining the color channels together to form RGB image
	high_pass_image = im2
	high_pass_image[:, :, 0] =  highpass_blue
	high_pass_image[:, :, 1] =  highpass_green
	high_pass_image[:, :, 2] =  highpass_red


	# kernel = np.array([[-1, -1, -1, -1, -1],
	#                    [-1,  1,  2,  1, -1],
	#                    [-1,  2,  4,  2, -1],
	#                    [-1,  1,  2,  1, -1],
	#                    [-1, -1, -1, -1, -1]])
	# highpass_red = ndimage.convolve(r2, kernel)
	# highpass_green = ndimage.convolve(g2, kernel)
	# highpass_blue = ndimage.convolve(b2, kernel)

	# red_lowpass = ndimage.gaussian_filter(highpass_red, 3)
	# red_gauss_highpass = r2 - red_lowpass

	# green_lowpass = ndimage.gaussian_filter(highpass_green, 3)
	# green_gauss_highpass = g2 - green_lowpass

	# blue_lowpass = ndimage.gaussian_filter(highpass_blue, 3)
	# blue_gauss_highpass = b2 - blue_lowpass

	# plt.figure()
	# plt.imshow(red_gauss_highpass)
	# plt.show()

	# green_lowpass = ndimage.gaussian_filter(im2_green_gaussian, 3)
	# green_highpass = im2_green_gaussian - green_lowpass

	# blue_lowpass = ndimage.gaussian_filter(im2_blue_gaussian, 3)
	# blue_highpass = im2_blue_gaussian - blue_lowpass

	# im2_red_sharpened = im2_red_gaussian + alpha * (im2_red_gaussian - im2_red_gaussian2)
	# im2_red_unsharpmask =  r2 *(im2_red_gaussian + 2*im2_red_sharpened - 2*im2_red_gaussian)

	# im2_green_sharpened = im2_green_gaussian + alpha * (im2_green_gaussian - im2_green_gaussian2)
	# im2_green_unsharpmask =  g2 *(im2_green_gaussian + 2*im2_green_sharpened - 2*im2_green_gaussian)


	# im2_blue_sharpened = im2_blue_gaussian + alpha * (im2_blue_gaussian - im2_blue_gaussian2)
	# im2_blue_unsharpmask =  b2 *(im2_blue_gaussian + 2*im2_blue_sharpened - 2*im2_blue_gaussian)
	# plt.figure()
	# plt.imshow(im2_blue_unsharpmask,cmap=plt.cm.gray)
	# plt.show()

	hybrid = low_pass_image + high_pass_image
	misc.imsave('../data/hybrid.png', hybrid)

	plt.figure()
	plt.imshow(hybrid)
	plt.show()


im1 = misc.imread('../data/dog.bmp')
im2 = misc.imread('../data/cat.bmp')

hybrid_img = hybrid_image(im1, im2)

	






