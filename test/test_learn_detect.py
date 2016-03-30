#!/usr/bin/env python

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
import cv2

from openface_ros.srv import LearnFace, DetectFace

bridge = CvBridge()

rospy.init_node('test_learn_detect')

learn_srv_name = "learn"
detect_srv_name = "detect"

rospy.loginfo("Waiting for services %s and %s" % (learn_srv_name, detect_srv_name))

rospy.wait_for_service(learn_srv_name)
rospy.wait_for_service(detect_srv_name)

learn_srv = rospy.ServiceProxy(learn_srv_name, LearnFace)
detect_srv = rospy.ServiceProxy(detect_srv_name, DetectFace)


def callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    cv2.imshow("Image window", cv_image)
    key = cv2.waitKey(10)

    if key == 1048684: # L
        print learn_srv(image=data, name=raw_input("Name? "))
    elif key == 1048676: # D
        print detect_srv(image=data)

    return

image_sub = rospy.Subscriber("image", Image, callback)
rospy.loginfo("Listening to %s -- spinning .." % image_sub.name)
rospy.loginfo("Usage: L to learn, D to detect")

rospy.spin()
