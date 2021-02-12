# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2


# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl,cnt_frame):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Open3D', width=1280, height=720, left=50, top=50, visible=True)
    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()
    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcl = pcl[:,:3]
    pcd.points = o3d.utility.Vector3dVector(pcl)
    

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    if cnt_frame == 0:
        print('add')
        vis.clear_geometries()
        vis.add_geometry(pcd)      
    else:
        print('update')
        vis.clear_geometries()
        vis.update_geometry(pcd)
           
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    def exit_key(vis):
        vis.destroy_window()
    
    vis.register_key_callback(262,exit_key)
    vis.poll_events()
    
    vis.run()

    #######
    ####### ID_S1_EX2 END #######     
    
# visualize range image
def show_range_image(frame):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar_name = dataset_pb2.LaserName.TOP
    lidar = waymo_utils.get(frame.lasers, lidar_name)
    range_image, camera_projection, range_image_pose = waymo_utils.parse_range_image_and_camera_projection(lidar)
    
    # step 2 : extract the range and the intensity channel from the range image
    range_image_range = range_image[...,0]
    range_image_intensity = range_image[...,1]
    
    # step 3 : set values <0 to zero
    range_image_range = np.where(range_image_range < 0, 0, range_image_range)
    range_image_intensity = np.where(range_image_intensity < 0, 0, range_image_intensity)
    
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    range_image_range = range_image_range.astype(np.int8)
    
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    range_image_intensity = range_image_intensity.astype(np.int8)
    mean, std = np.mean(range_image_intensity), np.std(range_image_intensity)
    ri_norm = (range_image_intensity-mean)/std
    
    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    ri= np.vstack((range_image_range, range_image_intensity))
    ri = ri.astype(np.int8)

    img_range_intensity = ri 
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]) &
                    (lidar_pcl[:, 3] >= configs.lim_r[0]) & (lidar_pcl[:, 3] <= configs.lim_r[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    x_lidar = lidar_pcl[:, 0]
    y_lidar = lidar_pcl[:, 1]
    z_lidar = lidar_pcl[:, 2]

    X =  configs.lim_x[0]
    x =  configs.lim_x[1]
    Y =  configs.lim_y[0]
    y =  configs.lim_y[1]
    Z =  configs.lim_z[0]
    z =  configs.lim_z[1]

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    dsc_x = (X - x)/(configs.bev_height-1)
    
    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:,0] = (y_lidar - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
    lidar_pcl_cpy[:,0] = np.floor(((lidar_pcl_cpy[:,0] - x)/dsc_x)).astype(np.int32)
    
    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    dsc_y = (Y - y)/(configs.bev_height-1)
    lidar_pcl_cpy[:,1] = (-x_lidar - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
    lidar_pcl_cpy[:,1] = np.floor(((lidar_pcl_cpy[:,1] - y)/dsc_y)).astype(np.int32)

    print(lidar_pcl_cpy.shape)
    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    show_pcl(lidar_pcl_cpy,0)
    
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    BEV = np.zeros(shape=(configs.bev_height,configs.bev_width,3))

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    #print(lidar_pcl_cpy.shape)
    
    ind = np.lexsort((-lidar_pcl_cpy[:,2],lidar_pcl_cpy[:,1],lidar_pcl_cpy[:,0]))

    ar = [lidar_pcl_cpy[ind[0],0], lidar_pcl_cpy[ind[0],1], lidar_pcl_cpy[ind[0],2]]
    
    for i in ind-1:
        x = lidar_pcl_cpy[ind[i],0]
        y = lidar_pcl_cpy[ind[i],1]
        z = lidar_pcl_cpy[ind[i],2]
        arr = [x,y,z]
        ar = np.vstack((ar,arr))

    print (ar.shape)
    
    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    unique,count = np.unique(ar[:,0:2],axis=0,return_counts=True)
    print (unique)

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    
    for i in unique:
        print(i)
        #Intensity Map
        BEV[i[0],i[1],1] =unique[i] #Choose the point with the heighest intensity
   
    intensity_map = BEV
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    cv2.imshow('intensity map', intensity_map)
    cv2.waitKey(0)
    #plt.imshow(BEV,cmap='cool', interpolation='nearest')
    #plt.show()
    
    #import test as tt
    #tt.birds_eye_point_cloud(lidar_pcl)
    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    BEV = np.zeros((3, configs.bev_height, configs.bev_width))

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    for i,c in zip(lidar_pcl_top.astype(np.int32),count):
        index = np.where((lidar_pcl_cpy[:,1] == i[1]))
        #Height Map
        BEV[i[0],i[1],2] = np.max(lidar_pcl_cpy[index][:,2]) #Choose the point with the heighest height
    
    height_map = BEV
    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    cv2.imshow('height map', height_map)
    cv2.waitKey(0)
    #######
    ####### ID_S2_EX3 END #######       
    
    
    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height, configs.bev_width))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) # used for model training, please do not change
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    
    return input_bev_maps


