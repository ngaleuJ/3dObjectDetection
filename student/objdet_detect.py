# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import cv2
import torch
from easydict import EasyDict as edict
import argparse

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing 
from tools.objdet_models.resnet.utils.torch_utils import _sigmoid
from tools.objdet_models.resnet.utils.evaluation_utils import time_synchronized

from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2

from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
import misc.objdet_tools as tools


# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()  

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))    
    
    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.iou_thresh = 0.5
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######     
        #######
        print("student task ID_S3_EX1-3")
        '''
        parser = argparse.ArgumentParser(description='Testing config for the Implementation')
        parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
        parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
        parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
        parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
        parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
        parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
        parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
        parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
        parser.add_argument('--peak_thresh', type=float, default=0.2)
        parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
        parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
        parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
        parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

        configs = edict(vars(parser.parse_args()))
        '''

        configs.saved_fn = 'fpm_resnet_18'
        configs.pretrained_path = '../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth'
        configs.no_cuda = False
        configs.gpu_idx = 0
        configs.num_samples = None
        configs.num_workers = 1
        configs.arch = 'fpn_resnet_18'
        configs.pin_memory = True
        configs.distributed = False  # For testing on 1 GPU only
        configs.K = 50
        configs.peak_thresh = 0.2
        configs.batch_size = 1
        configs.save_test_output = False
        configs.output_format = 'image'
        configs.output_video_fn = 'out_fpn_resnet_18'
        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50
        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2  # sin, cos
        

        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }
        configs.num_input_features = 4
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')

        

        #######
        ####### ID_S3_EX1-3 END #######     

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs


# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()    

    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50] # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0] # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608 

    # add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # visualization parameters
    configs.output_width = 608 # width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs


# create model according to selected model type
def create_model(configs):

    # check for availability of model file
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    # create model depending on architecture name
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
    elif 'fpn_resnet_18' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        
        ####### ID_S3_EX1-4 START #######     
        #######
        print("student task ID_S3_EX1-4")
        #print('using resnet')
        configs = load_configs('fpn_resnet')
        model = waymo_utils.create_model(configs)
        
        #######
        ####### ID_S3_EX1-4 END #######     
    
    else:
        assert False, 'Undefined model backbone'

    # load model weights
    
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))

    # set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    model.eval()          

    return model


# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():  

        # perform inference
        outputs = model(input_bev_maps)
        

        # decode model output into target object format
        if 'darknet' in configs.arch:

            # perform post-processing
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    

        elif 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing
            
            ####### ID_S3_EX1-5 START #######     
            #######
            
            print("student task ID_S3_EX1-5")
            #input_bev_maps = input_bev_maps.unsqueeze(0).to(configs.device, non_blocking=True).float()
            t1 = time_synchronized()
            #outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)

            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs)
            t2 = time_synchronized()

            detections = detections[0][1]  # Only vehicle class
            #print(detections)
            #######
            ####### ID_S3_EX1-5 END #######     

            

    ####### ID_S3_EX2 START #######     
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = [] 
    ## step 1 : check wether there are any detections
    if len(detections) > 0:

        ## step 2 : loop over all detections
        for row in detections:
            # extract detection
            _id, _x, _y, _z, _h, _w, _l, _yaw = row

            # convert from pixel into metric coordinates
            x = _y/configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
            y = _x/configs.bev_width * (configs.lim_y[1] - configs.lim_y[0]) - (configs.lim_y[1] - configs.lim_y[0])/2.0
            w = _w / configs.bev_width  * (configs.lim_y[1] - configs.lim_y[0])
            l = _l /configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
            #yaw = -_yaw

            ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
            if ((x >= configs.lim_x[0]) and (x <= configs.lim_x[1]) and (y >= configs.lim_y[0]) and (y <= configs.lim_y[1]) 
                and (_z >= configs.lim_z[0]) and (_z <= configs.lim_z[1])):

                ## step 4 : append the current object to the 'objects' array
                objects.append([1, x, y, _z, _h, w, l, _yaw])
        
                #tools.project_detections_into_bev(input_bev_maps,detections,configs,color=['red'])
    #######
    ####### ID_S3_EX2 START #######   
    
    return objects    

