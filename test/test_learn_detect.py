#!/usr/bin/env python

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
import cv2
import sys

from openface_ros.srv import LearnFace, DetectFace
from std_srvs.srv import Empty
from std_msgs.msg import String

bridge = CvBridge()

rospy.init_node('test_learn_detect')

try:
    interactive = rospy.get_param( "~interactive", True )
    external_api_request = rospy.get_param("~external_api_request", True)
    # save_images = rospy.get_param( "~save_images", True )
    max_distance_for_match = rospy.get_param("~max_distance_for_match", .66)
except KeyError as e:
    rospy.logerr("Please specify param: %s", e)
    sys.exit(1)

learn_srv_name = "learn"
detect_srv_name = "detect"
clear_srv_name = "clear"

rospy.loginfo("Waiting for services %s, %s and %s" % (learn_srv_name, detect_srv_name, clear_srv_name))

rospy.wait_for_service(learn_srv_name)
rospy.wait_for_service(detect_srv_name)
rospy.wait_for_service(clear_srv_name)

learn_srv = rospy.ServiceProxy(learn_srv_name, LearnFace)
detect_srv = rospy.ServiceProxy(detect_srv_name, DetectFace)
clear_srv = rospy.ServiceProxy(clear_srv_name, Empty)

def msg_pub(name):
    pub = rospy.Publisher('face_recognition_name', String, queue_size=10)
    pub.publish( name )

def callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    cv2.imshow("Image window", cv_image)
    key = cv2.waitKey(10)
    if ( not interactive ): # keep doing detection
        key = 1048676;

    if key == 1048684 or key == 108: # L
        print learn_srv(image=data, name=raw_input("Name? "))
    elif key == 1048676 or key == 100: # D
        detections = detect_srv(image=data, external_api_request=external_api_request)
        print detections

        # handle no detections
        if ( detections.face_detections.__len__() == 0):
            print "no detections"
            msg_pub( '(no detections)' )
            return
        
        # unpack detections, find the winner
        names = detections.face_detections[0].names
        distances = detections.face_detections[0].l2_distances

        min_distance = 1000.
        min_distance_idx = 1000.
        for idx, val in enumerate( distances ):
            if ( val < min_distance ):
                min_distance = val
                min_distance_idx = idx
        if min_distance < max_distance_for_match:
            match_name = names[ min_distance_idx ]
            print names[ min_distance_idx ] + ' with distance ' + str( min_distance )
        else:
            match_name = '(unknown)'
            print '(unknown)'

        # publish
        msg_pub( match_name )
    elif key == 1048675 or key == 99: # C
        print clear_srv()

    return

image_sub = rospy.Subscriber("image", Image, callback)
rospy.loginfo("Listening to %s -- spinning .." % image_sub.name)
rospy.loginfo("Usage: L to learn, D to detect")

rospy.spin()
