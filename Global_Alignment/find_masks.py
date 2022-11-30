import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import img_as_ubyte

filedir = 'C:\\Education\\University of Toronto\\Year 4\\Zhen Lab\\Z Alignment\\Dauer Test Images\\'
savedir = filedir
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

        ''''''
        for circle in circles[0, :]:
            a, b = int(circle[0]), int(circle[1])
            radius = int(circle[2])
            cv.circle(img=img1, center=(a, b), radius=radius, color=(255, 0, 0), 
                    thickness=2)
            cv.imshow('Circle Segment', img1), cv.waitKey(0), cv.destroyAllWindows()
        ''''''

        circle_array = np.zeros((500, 500), dtype = np.uint8)
        for circle in circles[0, :]:
            a, b = int(circle[0]), int(circle[1])
            radius = int(circle[2])
            cv.circle(img=circle_array, center=(a, b), radius=radius, color=255, 
                    thickness=-1)

        kernel = np.ones((5, 5), np.uint8)

        return cv.dilate(circle_array, kernel, iterations = 1)

    except:

        return find_mask(img, p2 - 10)
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

        elif len(circles[0, :]) > 9:

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

#if p2 is 0, no mask, skip for manual fixing
#run algo with p2
#if number of circles < 5 or > 10, redo
#higher p2 means less circles
#too many circles -> increase p2
#too little circles -> decrease p2

def load_img(index):

    return cv.imread(filedir + 'Region_' + str(index).zfill(4) + '_r1-c1.tif')

start = 20
end = 24

for i in range(start, end + 1):

    img = cv.resize(load_img(i), (500, 500))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    mask = find_mask(edges, 200)
    print(gray.shape)
    print(mask.shape)
    
    res = cv.addWeighted(gray,0.7,mask,0.3,0)
    
    cv.imwrite(savedir + f'mask_{i}.tif', mask)
    cv.imwrite(savedir + f'img_{i}.tif', res)

#try blob detection
#otherwise, just use canny with 10, 200 as params
#adjust param2 until the number of circles is between 5 and 10 maybe