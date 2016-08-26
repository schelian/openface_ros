# openface_ros
ROS wrapper for https://github.com/cmusatyalab/openface

# Setup
In addition to openface_ros, clone the following repositories from https://github.com/tue-robotics:
rgbd, tue_serialization, geolib2, code_profiler, ed_perception, ann2, tue_config, tue_filesystem, ed, ed_object_models, ed_sensor_integration, blackboard, cb_planner_msgs_srvs, and vocabulary_tree

Also clone usb_cam from https://github.com/bosch-ros-pkg/usb_cam.

Also create a symbolic link from where ever openface is installed to ~.

catkin build

You may have to build ed, then ed_perception before building the other packages (e.g., catkin build ed, catkin build ed_perception, etc.).

Also, if you have OpenCV or CUDA set up differently than usual set those as cmake arguments (e.g., catkin config -a --cmake-args -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-7.5 -DOpenCV_DIR=/home/<username>/local/share/OpenCV).

# How to test

    Terminal 1: 
        # start roscore & the usb camera node
        roscore &
        cd <location of workspace>
        source ./devel/setup.bash
        rosrun usb_cam usb_cam_node &

        # original use case: this will not load or save a library
        rosrun openface_ros openface_node.py &
        -or-
        # security use case: this will load a library if the file exists, otherwise it will create it; the library will be updated after learning
        rosrun openface_ros openface_node.py _face_dict_filename:=<my filname.pickle> &

        # echo messages from the openface node
        rostopic echo face_recognition_name

    Terminal 2:
        cd <location of workspace>
        source ./devel/setup.bash

        # original use case: this is interactive
        rosrun openface_ros test_learn_detect.py image:=/usb_cam/image_raw _external_api_request:=true
	-or-	
        # security use case: not interactive (always detects), does not use call the external api, and will not save out images (for speed)
        rosrun openface_ros test_learn_detect.py image:=/usb_cam/image_raw _external_api_request:=false _interactive:=false _save_images:=false

- Learn (L openCV Waitkey)
- Detect (D openCV Waitkey)
- Clear (C openCV Waitkey)
L, D, and C should be in the OpenCV window
