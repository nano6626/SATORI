import functools as ft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import cv2 as cv
from jax.lib import xla_bridge
from skimage import img_as_ubyte
PIL.Image.MAX_IMAGE_PIXELS = 400000000
print(xla_bridge.get_backend().platform)
count = cv.cuda.getCudaEnabledDeviceCount()
print(count)

from sofima.flow_field import JAXMaskedXCorrWithStatsCalculator
from sofima.flow_utils import clean_flow
from sofima.warp import ndimage_warp
from sofima import mesh
from itertools import combinations
from math import comb

#initial parameters

load_dir = '/mnt/onetb/yuuka/aligned_sections/'

def load_img(index):
    
    img = Image.open(load_dir + f'Fixed_Layer{index}.tif')
    
    return img

#finding possible pairs

def get_pairs(start, end, depth):
    
    pairs = []
    
    i = start
    while i < end:
        
        if end - i < depth:
            depth -= 1
        
        for j in range(1, depth + 1):
            pairs.append((i, i + j))
        
        i += 1
    
    return pairs

#given image, find a mask

def find_mask(image):
    
    img_ub = img_as_ubyte(image)
    
    edges = cv.Canny(img_ub, 0, 50)
    
    col, row = edges.shape
    
    num_of_rays = 10
    step = col//num_of_rays
    
    col_ind = np.arange(0, col, step)
    row_ind = np.arange(0, row, step)
    
    starts = []
    ends = []
    for i in row_ind:
        white_ind = np.where(edges[i,:] == 255)[0]
        if len(white_ind) == 0:
            continue
        else:
            starts.append(white_ind[0])
            ends.append(white_ind[-1])

    row_min = np.min(starts)
    row_max = np.max(ends)

    starts = []
    ends = []
    for j in col_ind:
        white_ind = np.where(edges[:,j] == 255)[0]
        if len(white_ind) == 0:
            continue
        else:
            starts.append(white_ind[0])
            ends.append(white_ind[-1])

    col_min = np.min(starts)
    col_max = np.max(ends)

    mask = np.zeros_like(img_ub)
    mask = cv.rectangle(mask, (row_min, col_min), (row_max, col_max), 255, -1)
    
    tlc = [row_min, col_min]
    brc = [row_max, col_max]
    
    masked_image = np.zeros((row_max - row_min, col_max - col_min))
    
    masked_image = image[col_min:col_max, row_min:row_max]
    
    return tlc, brc

def generate_mask(img, tlc, brc):
    
    mask = np.zeros_like(img)
    mask = cv.rectangle(mask, (tlc[0], tlc[0]), (brc[0], brc[1]), 255, -1)
    
    return mask

#setting up some useful functions

def choose_node_distribution(image, step):
    '''eats an image, returns mesh nodes'''
    
    step_row = step
    step_col = step
    
    row, col = image.shape
    
    col_ind = np.linspace(0, col-1, step_col)
    row_ind = np.linspace(0, row-1, step_row)
    
    nodes = []
    for i in col_ind:
        for j in row_ind:
            nodes.append([i,j]) #maybe use zip here

    nodes = np.array(nodes)
    
    return nodes

#this mandem finds the nbhd

def find_nbhd(target_img, x, y, epsilon):

    col, row = target_img.shape

    #there are 9 cases LOL
    #case 1, x - epsilon and y - epsilon is less than 0 ###
    #case 2, x - epsilon is less than 0 (only) ###
    #case 3, x - epsilon is less than 0 and y + epsilon > row ###
    #case 4, y + epsilon > row only ###
    #case 5, y + epsilon > row and x + epsilon > col ###
    #case 6, x + epsilon > col only ###
    #case 7, x + epsilon > col and y - epsilon less than 0 ###
    #case 8, y - epsilon less than 0 only
    #case 9, everything else

    if x < epsilon and y < epsilon:
        return target_img[0:x+epsilon, 0:y+epsilon], [0,0]
    
    elif x < epsilon and y + epsilon > row:
        return target_img[0:x+epsilon, y-epsilon:row], [0, y-epsilon]
    
    elif x < epsilon:
        return target_img[0:x+epsilon, y-epsilon:y+epsilon], [0, y-epsilon]

    elif x + epsilon > col and y < epsilon:
        return target_img[x-epsilon:col, 0:y+epsilon], [x-epsilon, 0]
    
    elif x + epsilon > col and y + epsilon > row:
        return target_img[x-epsilon:col, y-epsilon:row], [x-epsilon, y-epsilon]

    elif x + epsilon > col:
        return target_img[x-epsilon:col, y-epsilon:y+epsilon], [x-epsilon, y-epsilon]

    elif y + epsilon > row:
        return target_img[x-epsilon:x+epsilon, y-epsilon:row], [x-epsilon, y-epsilon]

    elif y < epsilon:
        return target_img[x-epsilon:x+epsilon, 0:y+epsilon], [x-epsilon, 0]

    else:
        return target_img[x-epsilon:x+epsilon, y-epsilon:y+epsilon], [x-epsilon, y-epsilon]
    
def correlation(img1, img2, patch_size, step, batch_size):
    maskcorr = JAXMaskedXCorrWithStatsCalculator()
    flow = maskcorr.flow_field(img1, img2, patch_size, step, batch_size = batch_size)
    return flow

def node_in_mask(node, source_mask):
    
    x = int(node[0])
    y = int(node[1])
    
    if source_mask[x,y] == 0:
        return False
    else:
        return True
    
def filters(offset_x, offset_y, pr, mesh_thresh, min_pr):
    '''not needed yet'''
    
    if np.isnan(offset_x[0]):
        return True
    
    elif np.sqrt(offset_x[0]**2 + offset_y[0]**2) > mesh_thresh:
        return True
    
    elif pr < min_pr:
        return True
    
    else:
        return False

#obtain all masks in compact form

def get_masks(start, end):
    
    mask_corners = []
    
    for i in range(start, end + 1):
        
        img = np.array(load_img(i).resize((5000,5000)))
        
        tlc, brc = find_mask(img)
        
        mask_corners.append([tlc, brc])
        
    return np.array(mask_corners)

#finding matching points between pairs

def pairwise_matching(img1, img2, mask_corners1, mask_corners2, nbhd_size, num_of_nodes, mesh_thresh, min_pr):
    
    nodes = choose_node_distribution(img1, num_of_nodes)
    
    mask1 = generate_mask(img1, mask_corners1[0], mask_corners1[1])
    mask2 = generate_mask(img2, mask_corners2[0], mask_corners2[1])
    
    source_target_pairs = []
    
    for node in nodes:
    
        if not node_in_mask(node, mask1):
            continue
        
        source_nbhd, raymoo = find_nbhd(img1, int(node[0]), int(node[1]), nbhd_size)
        target_nbhd, marisad = find_nbhd(img2, int(node[0]), int(node[1]), nbhd_size)

        offset_x, offset_y, pr, momobako = correlation(source_nbhd, target_nbhd, (nbhd_size, nbhd_size), nbhd_size * 4, 1)
        
        if np.isnan(offset_x[0]):
            continue

        elif np.sqrt(offset_x[0]**2 + offset_y[0]**2) > mesh_thresh:
            continue

        elif pr < min_pr:
            continue

        else:

            target_x = node[0] + offset_x[0]
            target_y = node[1] + offset_y[0]
            
            source_target_pair = np.array([[node[0], node[1]],[target_x[0], target_y[0]]])
            source_target_pairs.append(source_target_pair)
            
    source_target_pairs = np.array(source_target_pairs)            
    
    return source_target_pairs

#get matching points for all images

#start = 110
#end = 119
depth = 1

def find_matching_points(start, end, masks, num_of_nodes, nbhd_size, mesh_thresh, pr):
    
    matching_points = {}
    
    pairs = get_pairs(start, end, depth)
    print(pairs)
    
    for pair in pairs:
        
        img1 = np.array(load_img(pair[0]).resize((5000,5000)))
        img2 = np.array(load_img(pair[1]).resize((5000,5000)))
        
        mask_corners1 = masks[pair[0] - start]
        mask_corners2 = masks[pair[1] - start]
        
        matches = pairwise_matching(img1, img2, mask_corners1, mask_corners2, nbhd_size, num_of_nodes, mesh_thresh, pr)
        
        matching_points[pair[0], pair[1]] = matches
        
        print(pair)
    
    return matching_points

#masks = get_masks(start, end)
#mpts = find_matching_points(start, end)