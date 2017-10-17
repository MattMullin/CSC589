
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


def hybrid_Image_Gray(im1,im2):
	#read in the images and flatten them 
	#plane = misc.imread('../data/plane.bmp',flatten=1)
	#bird = misc.imread('../data/bird.bmp',flatten=1)

	#image_data = imread('../data/bird.bmp').astype(np.float32)
	#print 'Size: ', image_data.size
	#print 'Shape: ', image_data.shape



	gaussian_kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 36, 48, 36, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])
	im1_gaussian = filters.convolve(im1, gaussian_kernel, mode="mirror")
	im2_gaussian = filters.convolve(im2, gaussian_kernel, mode="mirror")

	#Saving each blurred image back to the data directory
	misc.imsave('im1_gaussian.png', im1_gaussian)
	misc.imsave('im2_gaussian.png', im2_gaussian)
	#plt.figure()
	#plt.imshow(plane_gaussian,cmap=plt.cm.gray)
	#plt.show()

	#this takes the blurred plane image and applies a mask to it which removes the high frequency
	#The mask is a square that is 20x20 in the middle of the frame
	p = np.fft.fft2(im1_gaussian)
	# fshift the FFT image.
	pshift = np.fft.fftshift(p)
	magnitude_spectrum = (np.abs(pshift))
	rows, cols = im1.shape
	crow, ccol = rows/2, cols/2  
	mask = np.zeros((rows, cols), np.uint8)
	mask[crow-12:crow+12, ccol-12:ccol+12] = 1
	#plt.figure()
	#plt.imshow(mask,cmap=plt.cm.gray)
	#plt.show()

	#creates a low pass image of the plane by convolving the blurred image with the mask
	low_pass_im1 = scipy.ndimage.filters.convolve(im1_gaussian,mask, mode='constant', cval=0.0)
	#plt.figure()
	#plt.imshow(low_pass_plane,cmap=plt.cm.gray)
	#plt.show()


	#Applying a second blur to the bird image
	im2_gaussian2 = filters.convolve(im2_gaussian, gaussian_kernel, mode="mirror")
	alpha = 0.1
	im2_sharpened = im2_gaussian + alpha * (im2_gaussian - im2_gaussian2)
	im2_unsharpmask =  im2 *(im2_gaussian + 2*im2_sharpened - 2*im2_gaussian)
	#plt.figure()
	#lt.imshow(bird_unsharpmask,cmap=plt.cm.gray)
	#plt.show()

	hybrid = low_pass_im1 + im2_unsharpmask
	misc.imsave('../data/hybrid.png', hybrid)
	plt.figure()
	plt.imshow(hybrid,cmap=plt.cm.gray)
	plt.show()

	return()






