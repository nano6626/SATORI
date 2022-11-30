import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import img_as_ubyte

filedir = 'C:\\Education\\University of Toronto\\Year 4\\Zhen Lab\\Z Alignment\\Dauer Test Images\\'
savedir = filedir
filename1 = filedir + 'Region_0020_r1-c1.tif'
filename2 = filedir + 'Region_0021_r1-c1.tif'

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

def affine_alignment(img1, img2, mask1, mask2):

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    empty_array_1 = np.zeros(img1_gray.shape[:2], dtype=np.uint8)
    empty_array_2 = np.zeros(img2_gray.shape[:2], dtype=np.uint8)

    row1, col1 = img1_gray.shape
    row2, col2 = img2_gray.shape

    #mask_1 = cv.rectangle(empty_array_1, (col1 - 100, row1), (col1, row1), (0), thickness = -1)
    #mask_2 = cv.rectangle(empty_array_2, (col2 - 100, row2), (col2, row2), (0), thickness = -1)
    #todo, create masks

    sift = cv.SIFT_create()

    kp1, d1 = sift.detectAndCompute(img1_gray, mask = mask1)
    kp2, d2 = sift.detectAndCompute(img2_gray, mask = mask2)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(d1, d2, k = 2)

    ratio = 0.8

    good_matches = []
    for i, j in matches:
        if i.distance < j.distance * ratio:
            good_matches.append([i])

    #img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv.imshow('img', img3)
    #k = cv.waitKey(0)

    source = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    destination = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    source = source
    destination = destination

    M, mask = cv.estimateAffine2D(source, destination)

    M[0,2] *= 30
    M[1,2] *= 30

    #dst1 = cv.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    #dst2 = cv.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))

    return M

#mask the images, mask the worm itself
#take union of circles, dilate the image so its thicker, this is now the image mask
#matching points, features, etc only searched for in this mask
#should have if number of circles small, flag it to check if mask is right
#only search for matching points within the mask
'''
def find_mask(img, p2):

    if p2 == 0:
        print('Cannot find mask')
        return False

    try:

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        filtered_image = img_as_ubyte(gray)
        edges = cv.Canny(filtered_image, 100, 200)

        circles = cv.HoughCircles(image=edges, method=cv.HOUGH_GRADIENT, dp=1,  
                           minDist=20, param1=200, param2=p2, minRadius=80, 
                           maxRadius=200)

        print(len(circles[0,:]), 'number of circles')
        
        if len(circles[0, :]) < 5:

            print('Too few circles')
            find_mask(img, p2 + 10)

        
        for circle in circles[0, :]:
            a, b = int(circle[0]), int(circle[1])
            radius = int(circle[2])
            cv.circle(img=img1, center=(a, b), radius=radius, color=(255, 0, 0), 
                    thickness=2)
            cv.imshow('Circle Segment', img1), cv.waitKey(0), cv.destroyAllWindows()
        

        circle_array = np.zeros((500, 500), dtype = np.uint8)
        for circle in circles[0, :]:
            a, b = int(circle[0]), int(circle[1])
            radius = int(circle[2])
            cv.circle(img=circle_array, center=(a, b), radius=radius, color=255, 
                    thickness=-1)

        kernel = np.ones((5, 5), np.uint8)

        return cv.dilate(circle_array, kernel, iterations = 1)

    except:

        find_mask(img, p2 - 10)
'''

def find_mask(edges, p2):

    if p2 == 0:
        print('Cannot find mask')
        return False

    try:

        circles = cv.HoughCircles(image=edges, method=cv.HOUGH_GRADIENT, dp=1,  
                    minDist=10, param1=200, param2=p2, minRadius=80, 
                    maxRadius=190)

        if len(circles[0, :]) < 5:

            print(len(circles[0, :]), 'Too few circles')
            return find_mask(edges, p2 - 1)

        elif len(circles[0, :]) > 10:

            print(len(circles[0, :]), 'Too many circles')
            return find_mask(edges, p2 + 1)

        else:

            print(len(circles[0, :]), 'circles detected')

            circle_array = np.zeros((500, 500), dtype = np.uint8)
            for circle in circles[0, :]:
                a, b = int(circle[0]), int(circle[1])
                radius = int(circle[2])
                cv.circle(img=circle_array, center=(a, b), radius=radius, color=255, 
                        thickness=-1)
        
            kernel = np.ones((5, 5), np.uint8)

            return cv.dilate(circle_array, kernel, iterations = 1)

    except:

        print('no circles detected, reducing p2')
        return find_mask(edges, p2 - 1)

'''
filedir = 'C:\\Education\\University of Toronto\\Year 4\\Zhen Lab\\Z Alignment\\Dauer Test Images\\'
savedir = filedir
filename1 = filedir + 'Region_0020_r1-c1.tif'
filename2 = filedir + 'Region_0021_r1-c1.tif'

img1_full = cv.imread(filename1)
img2_full = cv.imread(filename2)

img1 = cv.resize(img1_full, (500, 500))
img2 = cv.resize(img2_full, (500, 500))

mask1 = find_mask(img1, 60)
mask2 = find_mask(img2, 40)

M = affine_alignment(img1, img2, mask1, mask2)

dst1 = cv.warpAffine(img1_full, M, (img1_full.shape[1], img1_full.shape[0]))
dst2 = cv.warpAffine(img2_full, M, (img2_full.shape[1], img2_full.shape[0]))

#fig, axs = plt.subplots(2,2)
#axs[0,0].imshow(img1)
#axs[0,1].imshow(img2)
#axs[1,0].imshow(dst1)
#axs[1,1].imshow(dst2)
#plt.show()

cv.imwrite(savedir + f'temp1.tif', dst1)
cv.imwrite(savedir + 'temp2.tif', img2)
'''

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

    return cv.imread(filedir + 'Region_' + str(index).zfill(4) + '_r1-c1.tif')

def load_mask(index):

    return cv.imread(filedir + f'mask_{i}.tif')

start = 20
end = 24
affine_motions = []
masks_not_deformed = []

for i in range(start, end):

    if i == start:

        img1_full = load_img(i)
        img2_full = load_img(i + 1)

        img1 = cv.resize(img1_full, (500, 500))
        img2 = cv.resize(img2_full, (500, 500))

        mask1 = load_mask(i)
        mask2 = load_mask(i+1)

        mask1 = cv.cvtColor(mask1, cv.COLOR_BGR2GRAY)
        mask2 = cv.cvtColor(mask2, cv.COLOR_BGR2GRAY)

        mask1_ds = cv.resize(mask1, (500, 500))
        mask2_ds = cv.resize(mask2, (500, 500))

        M = affine_alignment(img2, img1, mask2_ds, mask1_ds)

        masks_not_deformed.append(mask1)
        masks_not_deformed.append(mask2)
        affine_motions.append(M)

    else:

        img1_full = img2_full
        img2_full = load_img(i + 1)

        img1 = cv.resize(img1_full, (500, 500))
        img2 = cv.resize(img2_full, (500, 500))

        mask1 = load_mask(i)
        mask2 = load_mask(i+1)

        mask1 = cv.cvtColor(mask1, cv.COLOR_BGR2GRAY)
        mask2 = cv.cvtColor(mask2, cv.COLOR_BGR2GRAY)

        mask1_ds = cv.resize(mask1, (500, 500))
        mask2_ds = cv.resize(mask2, (500, 500))

        M = affine_alignment(img2, img1, mask2_ds, mask1_ds)

        masks_not_deformed.append(mask2)
        affine_motions.append(M)

#something that deforms masks

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

print(affine_motions)

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




