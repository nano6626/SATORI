import numpy as np
import cv2 as cv
from scipy.interpolate import LinearNDInterpolator
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

mesh_dir = '/home/chandoki/projects/def-mzhen/chandoki/rough_alignment/'
load_dir = '/home/chandoki/scratch/stitched3/'
savedir = '/home/chandoki/scratch/batch2/'
start = 700

mesh = np.load(mesh_dir + 'relaxed_mesh.npy', allow_pickle=True)[()]

def matches_to_flow(nodes, offsets):
    
    mesh_els = 50
    
    x = nodes[:,0]
    y = nodes[:,1]
    
    x_off = offsets[:,0]
    y_off = offsets[:,1]

    #finding x flow field

    X = np.linspace(0, 20000, 20000)
    Y = np.linspace(0, 20000, 20000)

    X, Y = np.meshgrid(X, Y)

    interp1 = LinearNDInterpolator(list(zip(x, y)), x_off)
    X_OFF = interp1(X, Y)
    print('test1')
    
    #finding y flow field

    interp2 = LinearNDInterpolator(list(zip(x, y)), y_off)
    Y_OFF = interp2(X, Y)
    print('test2')

    return X, Y, X_OFF, Y_OFF

def load_img(index):
    
    img = Image.open(load_dir + f'Fixed_Fixed_Layer{index}.tif')
    
    return img

for i in range(len(mesh)):

    index = i + start
    
    mesh_nodes = mesh[i][0]
    tri_points = mesh[i][1]
    
    offsets = mesh_nodes - tri_points
    X, Y, X_OFF, Y_OFF = matches_to_flow(tri_points * 4, offsets * 4)

    map_x = np.add(X_OFF.T, X)
    print('moge')
    map_y = np.add(Y_OFF.T, Y)
    print('my beloved')
            
    MAP_X = map_x.astype('float32')
    MAP_Y = map_y.astype('float32')

    img = np.array(load_img(index))
    cv.imwrite(savedir + f'pre_ds_{i}.tif', img)
    cv.imwrite(savedir + f'pre_zoom_{i}.tif', img[2000:3000, 2000:3000])
    
    img = cv.remap(img, MAP_X, MAP_Y, cv.INTER_LINEAR)
    
    cv.imwrite(savedir + f'post_ds_{i}.tif', img)
    cv.imwrite(savedir + f'post_zoom_{i}.tif', img[2000:3000, 2000:3000])

