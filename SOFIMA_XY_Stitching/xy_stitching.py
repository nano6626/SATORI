import functools as ft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import cv2 as cv
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
import sys

from sofima import stitch_rigid
from sofima import stitch_elastic
from sofima import flow_utils
from sofima import mesh
from sofima import warp

def infinity_test(array):
  youmu = False
  for j in range(array.shape[2]):
    for k in range(array.shape[3]):
      if array[0,0,j,k] == np.inf or array[1,0,j,k] == np.inf:
        youmu = True
        break
  return youmu

def rigid_stitching(tile_space, tile_map, overlaps_xy, min_overlap):
  cx, cy = stitch_rigid.compute_coarse_offsets(tile_space, tile_map, overlaps_xy = overlaps_xy, min_overlap = min_overlap)
  if infinity_test(cx) == True:
    cx = stitch_rigid.interpolate_missing_offsets(cx, -1)
    cx = stitch_rigid.interpolate_missing_offsets(cx, -2)
  if infinity_test(cy) == True:
    cy = stitch_rigid.interpolate_missing_offsets(cy, -1)
    cy = stitch_rigid.interpolate_missing_offsets(cy, -2)
  
  coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, cy)
  
  return np.squeeze(cx), np.squeeze(cy), coarse_mesh

def elastic_stitching(tile_map, cx, cy, patch_size, stride, batch_size):
  fine_x, offsets_x = stitch_elastic.compute_flow_map(tile_map, cx, 0, patch_size = patch_size, stride=(stride, stride), batch_size=batch_size)
  fine_y, offsets_y = stitch_elastic.compute_flow_map(tile_map, cy, 1, patch_size = patch_size, stride=(stride, stride), batch_size=batch_size)
  return fine_x, offsets_x, fine_y, offsets_y

def flow_filtering(fine_x, fine_y):
  kwargs = {"min_peak_ratio": 1.1, "min_peak_sharpness": 1.1, "max_deviation": 5, "max_magnitude": 0}
  fine_x_temp = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :] for k, v in fine_x.items()}
  fine_y_temp = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :] for k, v in fine_y.items()}

  kwargs = {"min_patch_size": 10, "max_gradient": -1, "max_deviation": -1}
  fine_x_temp = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :] for k, v in fine_x_temp.items()}
  fine_y_temp = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :] for k, v in fine_y_temp.items()}

  return fine_x_temp, fine_y_temp

def aggregate(tile_map, cx, cy, coarse_mesh, stride, fine_x, offsets_x, fine_y, offsets_y):
  data_x = (cx, fine_x, offsets_x)
  data_y = (cy, fine_y, offsets_y)

  fx, fy, x, nbors, key_to_idx = stitch_elastic.aggregate_arrays(
      data_x, data_y, tile_map,
      coarse_mesh[:, 0, ...], stride=(stride, stride))
  
  return fx, fy, x, nbors, key_to_idx

def render_img(tile_map, stride, coord_map, key_to_idx):
  idx_to_key = {v: k for k, v in key_to_idx.items()}
  meshes = {idx_to_key[i]: np.array(x[:, i:i+1 :, :]) for i in range(x.shape[1])}
  stitched, mask = warp.render_tiles(tile_map, meshes, stride=(stride, stride))
  return stitched
  
def sofima(tile_space, tile_map, overlaps_xy, min_overlap, patch_size, stride, config):
  cx, cy, coarse_mesh = rigid_stitching(tile_space, tile_map, overlaps_xy, min_overlap = min_overlap)
  fine_x, offsets_x, fine_y, offsets_y = elastic_stitching(tile_map, cx, cy, patch_size, stride, batch_size = 5)
  fx, fy, x, nbors, key_to_idx = aggregate(tile_map, cx, cy, coarse_mesh, stride, fine_x, offsets_x, fine_y, offsets_y)
  def prev_fn(x):
    target_fn = ft.partial(stitch_elastic.compute_target_mesh, x=x, fx=fx, fy=fy)
    x = jax.vmap(target_fn)(nbors)
    return jnp.transpose(x, [1, 0, 2, 3])
  x, ekin, t = mesh.relax_mesh(x, None, config, prev_fn=prev_fn)
  return x, key_to_idx, stride 
  
stride1 = 20

config1 = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=0.01, k=0.1, stride=stride1,
                                num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                dt_max=100, prefer_orig_order=True,
                                start_cap=0.1, final_cap=10., remove_drift=True)

stride2 = 20

config2 = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=0.05, k=0.1, stride=stride2,
                                num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                dt_max=100, prefer_orig_order=True,
                                start_cap=0.1, final_cap=10., remove_drift=True)
                                
overlap_x = np.arange(1000,0,-100)
overlap_y = np.arange(500,0,-100)
								
A = []
for i in range(25):
  if i < 10:
    A.append('0' + str(i))
  else:
    A.append(str(i))

A = np.array(A).reshape(5,5)

B = np.zeros_like(A)
for i in range(A.shape[0]):
  for j in range(A.shape[1]):
    B[i,j] = A[4-i, 4-j]

value_errors = []
other_errors = []

save_dir = '/home/chandoki/scratch/stitched3/'

#image_list = np.arange(121)
#bad_images = [21, 24, 30, 32, 58, 61, 70, 71, 88, 90, 97, 99, 100, 102, 103, 109, 120]
#good_images = np.delete(image_list, bad_images)

#from timeit import default_timer as timer
#start_whole = timer()
for i in [int(sys.argv[1])]:
    try:
        #print("start try")
        tile_map = {}
        #start = timer()
        for y in range(A.shape[0]):
            for x in range(A.shape[1]):
                #start = timer()
                tile_id = B[y, x]
                if i < 10:
                    with open('/home/chandoki/scratch/sections_700_1200_raw/' + f'Liver_OnPoint_000{i}_{tile_id}.tif', 'rb') as fp:
                        img = Image.open(fp)
                        tile_map[(x, y)] = np.array(img)[::-1,::-1]
                        print('test2')
                elif 9 < i < 100:
                    with open('/home/chandoki/scratch/sections_700_1200_raw/' + f'Liver_OnPoint_00{i}_{tile_id}.tif', 'rb') as fp:
                        img = Image.open(fp)
                        tile_map[(x, y)] = np.array(img)[::-1,::-1]
                elif 99 < i < 1000:
                    with open('/home/chandoki/scratch/sections_700_1200_raw/' + f'Liver_OnPoint_0{i}_{tile_id}.tif', 'rb') as fp:
                        img = Image.open(fp)
                        tile_map[(x, y)] = np.array(img)[::-1,::-1]
                #end = timer()
                #print(end - start, "try load tile map iteration")

                elif i > 999:
                    with open('/home/chandoki/scratch/sections_700_1200_raw/' + f'Liver_OnPoint_{i}_{tile_id}.tif', 'rb') as fp:
                        img = Image.open(fp)
                        tile_map[(x, y)] = np.array(img)[::-1,::-1]
        #end = timer()
        #print(end - start, "try for double loop time")
        #start = timer()
        x, key_to_idx, stride = sofima((5,5), tile_map, (overlap_x, overlap_y), 1, (120,120), stride1, config1)
        #end = timer()
        #print(end - start, "try sofima")
        #start = timer()
        idx_to_key = {v: k for k, v in key_to_idx.items()}
        meshes = {idx_to_key[i]: np.array(x[:, i:i+1 :, :]) for i in range(x.shape[1])}
        stitched, mask = warp.render_tiles(tile_map, meshes, stride=(stride, stride), parallelism = 5)
        #end = timer()
        #print(end - start, "try render_img")
        cv.imwrite(save_dir + f'Fixed_Fixed_Layer{i}.tif', stitched[::-1,::-1])
        print(i, 'success, config 1')
                
    except ValueError as e:
        print(e)
        try:
            #print("start except")
            #start = timer()
            x, key_to_idx, stride = sofima((5,5), tile_map, (overlap_x, overlap_y), 1, (40, 40), stride2, config2)
            #end = timer()
            #print(end - start, "except sofima")
            #start = timer()
            idx_to_key = {v: k for k, v in key_to_idx.items()}
            meshes = {idx_to_key[i]: np.array(x[:, i:i+1 :, :]) for i in range(x.shape[1])}
            stitched, mask = warp.render_tiles(tile_map, meshes, stride=(stride, stride), parallelism = 5)
            #end = timer()
            #print(end - start, "except render_img")
            cv.imwrite(save_dir + f'Fixed_Fixed_Layer{i}.tif', stitched[::-1,::-1])
            print(i, 'success, config 2')
            
        except ValueError:
            value_errors.append(i)
            print(i, 'value error, config 1+2 failed')

        except:
            other_errors.append(i)
            print(i, 'other error, config 1+2 failed')
    except:
        other_errors.append(i)
        print(i, 'other error, config 1 only failed')
#start_end = timer()
#print(start_end - start_whole, "entire run time")
print('value errors', value_errors)
print('other errors', other_errors)
