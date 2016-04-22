#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError

from openface_ros.srv import LearnFace, DetectFace
from openface_ros.msg import FaceDetection

import numpy as np
import cv2
import os
from datetime import datetime

import dlib
import openface

from face_client import FaceClient


def _get_roi(bgr_image, detection):
    factor_x = 0.1
    factor_y = 0.2

    # Get the roi
    min_y = detection.top()
    max_y = detection.bottom()
    min_x = detection.left()
    max_x = detection.right()

    dx = max_x - min_x
    dy = max_y - min_y

    padding_x = int(factor_x * dx)
    padding_y = int(factor_y * dy)

    # Don't go out of bound
    min_y = max(0, min_y - padding_y)
    max_y = min(max_y + padding_y, bgr_image.shape[0]-1)
    min_x = max(0, min_x - padding_x)
    max_x = min(max_x + padding_x, bgr_image.shape[1]-1)

    return bgr_image[min_y:max_y, min_x:max_x]


def _get_min_l2_distance(vector_list_a, vector_b):
    return min([np.dot(vector_a - vector_b, vector_a - vector_b) for vector_a in vector_list_a])


class OpenfaceROS:
    def __init__(self, align_path, net_path, storage_folder):
        self._bridge = CvBridge()
        self._learn_srv = rospy.Service('learn', LearnFace, self._learn_face_srv)
        self._detect_srv = rospy.Service('detect', DetectFace, self._detect_face_srv)

        # Init align and net
        self._align = openface.AlignDlib(align_path)
        self._net = openface.TorchNeuralNet(net_path, imgDim=96, cuda=False)
        self._face_detector = dlib.get_frontal_face_detector()

        self._face_dict = {}  # Mapping from string to list of reps

        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)

        self._storage_folder = storage_folder

        # For attributes
        self._face_client = FaceClient('69efefc20c7f42d8af1f2646ce6742ec', '5fab420ca6cf4ff28e7780efcffadb6c')

    def _update_detection_with_recognition(self, detection):
        try:
            recognition_rep = self._get_rep(detection["roi_image"])
            detection["names"] = self._face_dict.keys()
            detection["l2_distances"] = [_get_min_l2_distance(reps, recognition_rep) for reps in self._face_dict.values()]
        except Exception as e:
            warn_msg = "Could not get representation of face image but detector found one: %s" % e
            rospy.logwarn(warn_msg)

        return detection

    def _update_detections_with_attributes(self, detections):

        buffers = [cv2.imencode('.jpg', d["roi"])[1].tostring() for d in detections]

        response = self._face_client.faces_recognize(buffers)

        import ipdb; ipdb.set_trace()

        return detections

    def _get_rep(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        bb = self._align.getLargestFaceBoundingBox(rgb_image)
        if bb is None:
            raise Exception("Unable to find a face in image")

        aligned_face = self._align.align(96, rgb_image, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Unable to align face bb image")

        return self._net.forward(aligned_face)

    def _learn_face_srv(self, req):
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            return {"error_msg": error_msg}

        now = datetime.now()
        cv2.imwrite("%s/%s_learn_%s.jpeg" % (self._storage_folder, now.strftime("%Y-%m-%d-%H-%M-%d-%f"), req.name), bgr_image)

        try:
            rep = self._get_rep(bgr_image)
        except Exception as e:
            error_msg = "Could not get representation of face image: %s" % e
            rospy.logerr(error_msg)
            return {"error_msg": error_msg}

        if req.name not in self._face_dict:
            self._face_dict[req.name] = []

        self._face_dict[req.name].append(rep)

        rospy.loginfo("Succesfully learned face of '%s'" % req.name)

        return {"error_msg": ""}

    def _save_images(self, rois, bgr_image):
        now = datetime.now()
        cv2.imwrite("%s/%s_detect.jpeg" % (self._storage_folder, now.strftime("%Y-%m-%d-%H-%M-%d-%f")), bgr_image)

        for roi in rois:
            now = datetime.now()
            cv2.imwrite("%s/roi_%s_detection.jpeg" % (self._storage_folder, now.strftime("%Y-%m-%d-%H-%M-%d-%f")), roi)

    def _detect_face_srv(self, req):
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            return {"error_msg": error_msg}

        # Display the resulting frame
        detections = [{"roi": _get_roi(bgr_image, d), "x": d.left(), "y": d.top(),
                       "width": d.width(), "height": d.height()} for d
                      in self._face_detector(bgr_image, 1)]  # 1 = upsample factor

        self._save_images([d["roi"] for d in detections], bgr_image)

        # Try to recognize
        detections = [self._update_detection_with_recognition(d) for d in detections]

        # Try to add attributes
        detections = self._update_detections_with_attributes(detections)

        return {
            "face_detections": [FaceDetection(names=d["names"], l2_distances=d["l2_distances"],
                                              x=d["x"], y=d["y"], width=d["width"], height=d["height"],
                                              gender_is_male=d["attrs"]["gender_is_male"],
                                              gender_confidence=d["attrs"]["gender_confidence"], age=d["attrs"]["age"])
                                for d in detections],
            "error_msg": ""
        }

if __name__ == '__main__':
    rospy.init_node('openface')

    align_path_param = rospy.get_param('~align_path', os.path.expanduser('~/openface/models/dlib/shape_predictor_68_face_landmarks.dat'))
    net_path_param = rospy.get_param('~net_path', os.path.expanduser('~/openface/models/openface/nn4.small2.v1.t7'))
    storage_folder_param = rospy.get_param('~storage_folder', os.path.expanduser('/tmp/faces'))

    openface_node = OpenfaceROS(align_path_param, net_path_param, storage_folder_param)
    rospy.spin()
