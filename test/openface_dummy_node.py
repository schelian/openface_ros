#!/usr/bin/env python

import rospy
import os

from openface_ros.srv import LearnFace, DetectFace
from openface_ros.msg import FaceDetection

class OpenfaceROS:
    def __init__(self, align_path, net_path):
        self._learn_srv = rospy.Service('learn', LearnFace, self._learn_face_srv)
        self._detect_srv = rospy.Service('detect', DetectFace, self._detect_face_srv)

    def _learn_face_srv(self, req):
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            return {"error_msg": error_msg}

        rospy.loginfo("Succesfully learned face of '%s'" % req.name)

        return {"error_msg": ""}

    def _detect_face_srv(self, req):
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            return {"error_msg": error_msg}

        response = {
            "face_detections" : [],
            "error_msg" : ""
        }

        return response

if __name__ == '__main__':
    rospy.init_node('openface')

    align_path_param = rospy.get_param('~align_path', os.path.expanduser('~/openface/models/dlib/shape_predictor_68_face_landmarks.dat'))
    net_path_param = rospy.get_param('~net_path', os.path.expanduser('~/openface/models/openface/nn4.small2.v1.t7'))

    openface_node = OpenfaceROS(align_path_param, net_path_param)
    rospy.spin()
