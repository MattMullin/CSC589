from my_imfilter_color import hybrid_Image_Color
from my_imfilter import hybrid_Image_Gray
from scipy import misc

im1 = misc.imread('../data/dog.bmp')
im2 = misc.imread('../data/cat.bmp')

color_image = hybrid_Image_Color(im1,im2)


im3 = misc.imread('../data/fish.bmp',flatten=1)
im4 = misc.imread('../data/submarine.bmp',flatten=1)

gray_image = hybrid_Image_Gray(im3,im4)