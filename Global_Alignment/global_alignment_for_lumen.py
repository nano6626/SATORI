import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import img_as_ubyte

filedir = 'C:\\Education\\University of Toronto\\Year 4\\Zhen Lab\\Z Alignment\\Dauer Test Images\\'
savedir = filedir

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def affine_alignment(img1, img2):

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    empty_array_1 = np.zeros(img1_gray.shape[:2], dtype=np.uint8)
    empty_array_2 = np.zeros(img2_gray.shape[:2], dtype=np.uint8)

    row1, col1 = img1_gray.shape
    row2, col2 = img2_gray.shape

    sift = cv.SIFT_create()

    kp1, d1 = sift.detectAndCompute(img1_gray)
    kp2, d2 = sift.detectAndCompute(img2_gray)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(d1, d2, k = 2)

    ratio = 0.8

    good_matches = []
    for i, j in matches:
        if i.distance < j.distance * ratio:
            good_matches.append([i])

    source = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    destination = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    source = source
    destination = destination

    M, mask = cv.estimateAffine2D(source, destination)

    return M

#have N images
#find map between neighbouring images M from i+1 to i image
#apply maps using compositions
#can also have map aligned to previous image, this method sucks 
#deformations in previous images lead to even worse deformations in new images
#also it seems to be worse when doing masking on deformed images
#instead, store array of all transformations
#at the end, apply transformations to both images and masks
#save the masks, save the images

filedir = 'C:\\Education\\University of Toronto\\Year 4\\Zhen Lab\\Z Alignment\\Dauer Test Images\\'
savedir = filedir

def load_img(index):

    return cv.imread(filedir + 'Lumen_' + str(index) + '.tif')

start = 20
end = 24
affine_motions = []

for i in range(start, end):

    if i == start:

        img1_full = load_img(i)
        img2_full = load_img(i + 1)

        #img1 = cv.resize(img1_full, (500, 500))
        #img2 = cv.resize(img2_full, (500, 500))
        #optional downsampling, not necessary for lumen dataset

        mask1 = cv.cvtColor(mask1, cv.COLOR_BGR2GRAY)
        mask2 = cv.cvtColor(mask2, cv.COLOR_BGR2GRAY)

        M = affine_alignment(img2_full, img1_full)

        affine_motions.append(M)

    else:

        img1_full = img2_full
        img2_full = load_img(i + 1)

        #img1 = cv.resize(img1_full, (500, 500))
        #img2 = cv.resize(img2_full, (500, 500))

        mask1 = cv.cvtColor(mask1, cv.COLOR_BGR2GRAY)
        mask2 = cv.cvtColor(mask2, cv.COLOR_BGR2GRAY)

        mask1_ds = cv.resize(mask1, (500, 500))
        mask2_ds = cv.resize(mask2, (500, 500))

        M = affine_alignment(img2_full, img1_full)

        affine_motions.append(M)

#something that deforms images

def compose_affine_maps(M1, M2):
    #composition of M2 circ M1

    M1_new = np.zeros((3,3))
    M2_new = np.zeros((3,3))

    M1_new[0:2, :] = M1
    M1_new[2,2] = 1
    M2_new[0:2, :] = M2
    M2_new[2,2] = 1

    return np.matmul(M1_new, M2_new)[0:2, :]

#apply affine motions

start = 20
end = 24
prev_map = None

for i in range(start, end):

    if prev_map is None:
        
        img_full = load_img(i+1)
        #img_full = cv.resize(img_full, (500, 500))
        M = affine_motions[i - start]
        prev_map = M
        dst = cv.warpAffine(img_full, M, (img_full.shape[1], img_full.shape[0]))
        cv.imwrite(savedir + f'dauer_{i+1}.tif', dst)

    else:

        img_full = load_img(i+1)
        #img_full = cv.resize(img_full, (500, 500))
        M = compose_affine_maps(prev_map, affine_motions[i - start])
        prev_map = M
        dst = cv.warpAffine(img_full, M, (img_full.shape[1], img_full.shape[0]))
        cv.imwrite(savedir + f'dauer_{i+1}.tif', dst)