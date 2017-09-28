import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage


#reading in the images used throughout the code
cheetah = cv2.imread('cheetah.png')
zebra = cv2.imread('zebra.png')
smartman = cv2.imread('einstein.png', 0)


#Creating a Gaussian Blur over the two images with a 7x7 convolution kernal
cheetahblurred = cv2.GaussianBlur(cheetah, (7,7),0)
zebrablurred = cv2.GaussianBlur(zebra, (7,7),0)


#This is the method for plotting both of the blurred images
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(cheetahblurred, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Gaussian Cheetah')

ax2.imshow(zebrablurred, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Gaussian Zebra')
fig.tight_layout()

plt.show()

"""
canvas = np.zeros((300, 300, 3))
np.fft.ifft2(canvas)
a = np.mgrid[:5, :5][0]
np.fft.fft2(a)
"""

#Problem 2
#the low contrast images is read in
lowConImage = cv2.imread("lowcontrast.jpg")

#the image is shown and given a command to remain on the screen until another key is hit
cv2.imshow("Low Contrast Image", lowConImage)
cv2.waitKey()
cv2.destroyAllWindows()

#cv.split breaks the image into its three color channels, which in this case are arranged in the order of blue, green, red
chans = cv2.split(lowConImage)

#The three channels are converted in the following order: blue, green, red 
colors = ("b", "g", "r")
plt.figure()
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color = color)
	plt.xlim([0, 256])

# Show our plots
plt.show()



#Problem 3
#These are the equations for processing different filters using opencvs
#cv2.boxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]])
#cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
#cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
#ndimage.convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0)

w = np.array([[1,1,1],[1,2,1],[1,1,1]])

laplacian = cv2.Laplacian(smartman,cv2.CV_64F)
sobelx = cv2.Sobel(smartman,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(smartman,cv2.CV_64F,0,1,ksize=5)
gaussianBlur = cv2.GaussianBlur(smartman, (5, 5), 0)

sobelxd = ndimage.convolve(sobelx, w, mode='constant', cval=0.0)
sobelyd = ndimage.convolve(sobely, w, mode='constant', cval=0.0)


plt.subplot(3,2,1),plt.imshow(smartman,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(gaussianBlur,cmap = 'gray')
plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(sobelxd,cmap = 'gray')
plt.title('Sobel X Convolved'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(sobelyd,cmap = 'gray')
plt.title('Sobel Y Convolved'), plt.xticks([]), plt.yticks([])
plt.show()




#Problem number 4
from skimage import feature
from scipy import ndimage as ndi

# Generating the square 
square = np.zeros((150, 150))
square[50:-50, 50:-50] = 1

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(square)
edges2 = feature.canny(square, sigma=3)

# display results
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(square, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('White Square', fontsize=16)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Edge using canny filter', fontsize=16)
fig.tight_layout()

plt.show()


