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

    def _get_attrs(self, bgr_roi):
        buffer = cv2.imencode('.jpg', bgr_roi)[1].tostring()

        result = {
            "age" : 0,
            "gender_is_male" : False,
            "gender_confidence" : 0
        }

        try:
            response = self._face_client.faces_recognize('guido', buffer=buffer, namespace = 'robocup')
            attributes = response["photos"][0]["tags"][0]["attributes"]

            result["gender_is_male"] = attributes["gender"]["value"] == "male"
            result["gender_confidence"] = float(.01 * attributes["gender"]["confidence"])
            result["age"] = int(attributes["age_est"]["value"])
        except Exception as e:
            pass

        return result


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

    def _detect_face_srv(self, req):
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            return {"error_msg": error_msg}

        # Display the resulting frame
        detections = self._face_detector(bgr_image, 1)  # 1 = upsample factor

        response = {
            "face_detections" : [],
            "error_msg" : ""
        }

        now = datetime.now()
        cv2.imwrite("%s/%s_detect.jpeg" % (self._storage_folder, now.strftime("%Y-%m-%d-%H-%M-%d-%f")), bgr_image)

        for detection in detections:
            bgr_roi = bgr_image[detection.top():detection.bottom(), detection.left():detection.right()]

            now = datetime.now()
            cv2.imwrite("%s/%s_detection.jpeg" % (self._storage_folder, now.strftime("%Y-%m-%d-%H-%M-%d-%f")), bgr_roi)

            try:
                detection_rep = self._get_rep(bgr_roi)
            except Exception as e:
                warn_msg = "Could not get representation of face image: %s" % e
                rospy.logwarn(warn_msg)
                continue

            attrs = self._get_attrs(bgr_roi)

            response["face_detections"].append(FaceDetection(
                names=self._face_dict.keys(),
                l2_distances=[_get_min_l2_distance(reps, detection_rep) for reps in self._face_dict.values()],
                x=detection.top(),
                y=detection.left(),
                width=detection.width(),
                height=detection.height(),
                gender_is_male=attrs["gender_is_male"],
                gender_confidence=attrs["gender_confidence"],
                age=attrs["age"]
            ))

        return response

if __name__ == '__main__':
    rospy.init_node('openface')

    align_path_param = rospy.get_param('~align_path', os.path.expanduser('~/openface/models/dlib/shape_predictor_68_face_landmarks.dat'))
    net_path_param = rospy.get_param('~net_path', os.path.expanduser('~/openface/models/openface/nn4.small2.v1.t7'))
    storage_folder_param = rospy.get_param('~storage_folder', os.path.expanduser('/tmp/faces'))

    openface_node = OpenfaceROS(align_path_param, net_path_param, storage_folder_param)
    rospy.spin()
