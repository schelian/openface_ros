# openface_ros
ROS wrapper for https://github.com/cmusatyalab/openface

# How to test

    T1: rosrun usb_cam usb_cam_node
    T2: rosrun openface_ros openface_node.py
    T3: rosrun openface_ros test_learn_detect.py image:=/usb_cam/image_raw _external_api_request:=true

- Learn (L openCV Waitkey); type name as raw_input python
- Detect (D openCV Waitkey)
- Clear (C openCV Waitkey)
