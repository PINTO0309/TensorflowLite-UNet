######################################################
# Segmentation image padding program
######################################################

import cv2
import os
import glob
import numpy as np
from PIL import Image, ImageOps


######################################################
increase_num = 20
######################################################


# Histogram homogenization function
def equalizeHistRGB(src):
    
    RGB = cv2.split(src)
    Blue   = RGB[0]
    Green = RGB[1]
    Red    = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])
    img_hist = cv2.merge([RGB[0],RGB[1], RGB[2]])
    return img_hist

# Gaussian noise function
def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    var = 0.1
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss
    return noisy

# Salt & Pepper noise function
def addSaltPepperNoise(src):
    row,col,ch = src.shape
    s_vs_p = 0.5
    amount = 0.004
    out = src.copy()

    # Salt mode
    try:
        num_salt = np.ceil(amount * src.size * s_vs_p)
        coords = [np.random.randint(0, i-1 , int(num_salt)) for i in src.shape]
        out[coords[:-1]] = (255,255,255)
    except:
        pass

    # Pepper mode
    try:
        num_pepper = np.ceil(amount* src.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in src.shape]
        out[coords[:-1]] = (0,0,0)
    except:
        pass
    return out

# Rotation
def rotate_image(src1, src2, angle):
    orig_h, orig_w = src1.shape[:2]
    #matrix = cv2.getRotationMatrix2D((orig_h/2, orig_w/2), angle, 1)
    matrix = cv2.getRotationMatrix2D((orig_w/2, orig_h/2), angle, 1)
    return cv2.warpAffine(src1, matrix, (orig_w, orig_h), src1, flags=cv2.INTER_LINEAR), src2.rotate(angle)


img_filesJ = sorted(glob.glob("data_set/VOCdevkit/person/JPEGImages/*"))
img_filesS = sorted(glob.glob("data_set/VOCdevkit/person/SegmentationClass/*"))
JPEG_out_base_path = "data_set/VOCdevkit/person/JPEGImagesOUT"
SEGM_out_base_path = "data_set/VOCdevkit/person/SegmentationClassOUT"

imgs = []
for (img_fileJ, img_fileS) in zip(img_filesJ, img_filesS):
    imgs.append([cv2.imread(img_fileJ, cv2.IMREAD_UNCHANGED), Image.open(img_fileS)])

# Generate lookup table
min_table = 50
max_table = 205
diff_table = max_table - min_table
gamma1 = 0.75
gamma2 = 1.5
LUT_HC = np.arange(256, dtype = 'uint8')
LUT_LC = np.arange(256, dtype = 'uint8')
LUT_G1 = np.arange(256, dtype = 'uint8')
LUT_G2 = np.arange(256, dtype = 'uint8')
LUTs = []
# Smoothing sequence
average_square = (10,10)
# Create high contrast LUT
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table                        
for i in range(max_table, 255):
    LUT_HC[i] = 255
# Other LUT creation
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1) 
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
LUTs.append(LUT_HC)
LUTs.append(LUT_LC)
LUTs.append(LUT_G1)
LUTs.append(LUT_G2)

imgcnt = 0

for img in imgs:

    for i in range(increase_num):

        jpgimg = img[0]
        segimg = img[1]

#        # Contrast conversion execution
#        if np.random.randint(2) == 1:
#            level = np.random.randint(4)
#            jpgimg = cv2.LUT(jpgimg, LUTs[level])

        # Smoothing execution
        if np.random.randint(2) == 1:
            jpgimg = cv2.blur(jpgimg, average_square)

#        # Histogram equalization execution
#        if np.random.randint(2) == 1:
#            jpgimg = equalizeHistRGB(jpgimg)

        # Gaussian noise addition execution
        if np.random.randint(2) == 1:
            jpgimg = addGaussianNoise(jpgimg)

        # Salt & Pepper noise addition execution
        if np.random.randint(2) == 1:
            jpgimg = addSaltPepperNoise(jpgimg)

        # Rotation
        if np.random.randint(2) == 1:
            jpgimg, segimg = rotate_image(jpgimg, segimg, np.random.randint(360))

        # Reverse execution
        if np.random.randint(2) == 1:
            jpgimg = cv2.flip(jpgimg, 1)
            segimg = ImageOps.mirror(segimg)


        # Image storage after padding
        JPEG_image_path = "%s/%04d_%04d.jpg" % (JPEG_out_base_path, imgcnt, i)
        SEGM_image_path = "%s/%04d_%04d.png" % (SEGM_out_base_path, imgcnt, i)
        cv2.imwrite(JPEG_image_path, jpgimg)
        segimg.save(SEGM_image_path)

        print("imgcnt =", imgcnt, "num =", i)
    imgcnt += 1

print("Finish!!")
