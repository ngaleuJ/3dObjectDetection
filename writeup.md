# Writeup: Track 3D-Objects Over Time

Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?

* The four tracking steps for this project included: 
	1. ## Computing lidar point-cloud from range image.
		* This section consisted of converting a range image(RI) to a point cloud(PCL) then visualing it. Below are two examples of point cloud images with varying degrees of visibility.
	
		![frame_53](.\img\frame_53.png)
				*This is frame 53 of the orignal range image from the first sub dataset of the waymo dataset. From this, we are able to identify atleast 6 vehicles and distinct features like; tail lights, rear bumper and even the sides of the vehicles.*

		![frame_59_normal.png](.\img\frame_59_normal.png)
        		*This is frame 59 of the orignal range image from the first sub dataset of the waymo dataset. From this, we can clearly identify at least 6 vehicles.Zooming in the pcl using the viewer as you can see below, we are able to distinguish another distinct feature such as wheels*
        	
        ![frame_59_zoomed.png](.\img\frame_59_zoomed.png)
        
        ![frame_0.png](.\img\frame_0.png)
        	*This is frame 0 of the third sub dataset of the waymo data set. From this we are able to identify at least 10 vehicles with varying degrees of visibility. Zooming in as we can see below, we can identify more distinct features some have large number of points and others only a few*
        
		![frame_0_zoomed.png](.\img\frame_0_zoomed.png)
        	*From this PCL, we are able to identify more features like the windshield, front bumper and even the rear bumper of a truck. We can also distinguish the direction of movement of the vehicles. Oncoming and same driving direction.*

	2. ## Creating birds-eye view from lidar point-cloud.
		* In this section, we created a bird eyed view map(BEV) based on intensity, height and density from the PCL images we saw above. Below are examples of an intensity and height map we obtained. 
			![intensity_height_frame_53.png](.\img\intensity_height_frame_53.png)
            	*Intensity and height maps for frame 53*
            ![intensity_height_frame_59.png](.\img\intensity_height_frame_59.png)
            	*Intensity and height maps for frame 59*
			![intensity_PCL_height_frame_59.png](.\img\intensity_PCL_height_frame_59.png)
            	*Intensity,PCL and height maps for frame 59*

	3. ## Model Based object detection in the bird-eye view image.
		* Here, a pretrained model obtained from the *Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds* is used to perfom detection on the images displayed above. This is mainly done on frame 50 to 150.
		* After runing the fpn resnet model, this is the detection obtained. 
			
		![resnetrun.png](.\img\resnetrun.png)

		![ModelBasedOD.png](.\img\ModelBasedOD.png)

	4. ## Performance evaluation for object detection. 
		* We used the Intersection over union(IOU) we computed here in order to find pairings between the ground-truth labels and dectections. This was to be able to determine wether an object has been *(a) missed (false negative), (b) successfully detected (true positive) or (c) has been falsely reported (false positive)*.We obtained results such as
		![Figure_1detect 150frames.png](.\img\Figure_1detect 150frames.png)
        	*TP = 257, FP = 3, FN = 49
			precision = 0.9884615384615385, recall = 0.8398692810457516*


### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 
	* Yes, Using camera-lidar fusion ia more beneficial because it can encompass more information such as identifying traffic signs and light and help to take even better decisions.


### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
	* In case of very bad weather conditions, it will be very difficult to perform detections, In the scope of this project, I did not notice any of this.


### 4. Can you think of ways to improve your tracking results in the future?
	* Yes, by fine tuning the threshold for the iou and be more efficient in eliminating outliers in order to produce more accurate detections, 

