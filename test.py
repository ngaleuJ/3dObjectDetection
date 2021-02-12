from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
import misc.objdet_tools as objtools

import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy
import open3d as o3d

# Add current working directory to path
sys.path.append(os.getcwd())
vis_pause_time = 0
### FIRST TASK
# Select Waymo Open Dataset file and frame numbers
#data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'  # Sequence 1
data_filename ='training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord'

# adjustable path in case this script is called from another working directory
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename)
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)  # initialize dataset iterator
frame = next(datafile_iter)

#print (datafile)
print('##########################')
# print(frame.lasers[1])

'''
# extract lidar data and range image
lidar_name = dataset_pb2.LaserName.TOP
lidar = waymo_utils.get(frame.lasers, lidar_name)
# Parse the top laser range image and get the associated projection.
range_image, camera_projection, range_image_pose = waymo_utils.parse_range_image_and_camera_projection(lidar)
range_image_range = range_image[...,0]
range_image_intensity = range_image[...,1]
print(range_image_intensity.dtype)
range_image_range = np.where(range_image_range < 0, 0, range_image_range)
range_image_intensity = np.where(range_image_intensity < 0, 0, range_image_intensity)
#print(range_image_intensity)
range_image_range = range_image_range.astype(np.int8)
print(range_image_range.shape)
range_image_intensity = range_image_intensity.astype(np.int8)
mean, std = np.mean(range_image_intensity), np.std(range_image_intensity)
print('mean=%.3f stdv=%.3f' % (mean, std))
ri_norm = (range_image_intensity-mean)/std
ri= np.vstack((range_image_range, range_image_intensity))
cv2.imshow('range_image', ri)
cv2.waitKey(vis_pause_time)
'''


#### SECOND TASK

'''
pcd = o3d.geometry.PointCloud()
pcl = objtools.pcl_from_range_image(frame)
pcl = pcl[:,:3]
print(pcl.shape)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='Open3D', width=1920, height=1080, left=50, top=50, visible=True)

pcd.points = o3d.utility.Vector3dVector(pcl)
show_only_frames = [0, 1]

if show_only_frames[0] == 0:
    print('add')
    vis.add_geometry(pcd)
else:
    print('update')
    vis.update_geometry(pcd)
    
def exit_key(vis):
    vis.destroy_window()
    
vis.register_key_callback(262,exit_key)
vis.poll_events()            
vis.run() 

a = [4,5,6]
b = [1,2,3]

#a.append(b)
arr = np.vstack((a,b))
arr = np.vstack((arr,arr))
print (arr)
print (arr.shape)

'''
from PIL import Image
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

pcd = objtools.pcl_from_range_image(frame)
def birds_eye_point_cloud(points,
                          side_range=(-10, 10),
                          fwd_range=(-10,10),
                          res=0.1,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto=None):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[-y_img, x_img] = pixel_values # -y because images start from top left

    # Convert from numpy array to a PIL image
    im = Image.fromarray(im)

    im.show()


#birds_eye_point_cloud(pcd)