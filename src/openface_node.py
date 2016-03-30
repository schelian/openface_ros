#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from openface_ros.srv import LearnFace, DetectFace

class OpenfaceROS:

	def __init__(self):
		self._bridge = CvBridge()
		self._learn_srv = rospy.Service('learn', LearnFace, self._learn_face_srv)
		self._detect_srv = rospy.Service('detect', DetectFace, self._detect_face_srv)

	def _learn_face_srv(req):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	def _detect_face_srv(req):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	def callback(self, data):
		

		(rows,cols,channels) = cv_image.shape
		if cols > 60 and rows > 60 :
		cv2.circle(cv_image, (50,50), 10, 255)

		cv2.imshow("Image window", cv_image)
		cv2.waitKey(3)

		try:
		self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
	print(e)

if __name__ == '__main__':
    openface_node = OpenfaceROS()
	rospy.init_node('openface')
	rospy.spin()

__author__ = 'amigo'

# import os
# import collections
# import numpy as np
# import cv2
# from sklearn.svm import SVC
# import openface

# fileDir = os.path.dirname(os.path.expanduser("~/openface/demos/"))
# modelDir = os.path.join(fileDir, '..', 'models')
# dlibModelDir = os.path.join(modelDir, 'dlib')
# openfaceModelDir = os.path.join(modelDir, 'openface')

# imgDim = 96

# align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
# net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), imgDim)

# class FaceRecognizer(object):
#     def __init__(self):
#         self.traindata = []
#         self.trainlabels = []

#         self.classifier = SVC(C=1, kernel='linear', probability=True)

#     def add_sample(self, name, imagePath):
#         representation = self._represent(imagePath)

#         self.traindata += [representation]
#         self.trainlabels += [name]


#     def cluster(self):
#         self.classifier.fit(self.traindata, self.trainlabels)

#     def _represent(self, imagePath):
#         bgrImg = cv2.imread(imagePath)
#         if bgrImg is None:
#             raise Exception("Unable to load image: {}".format(imagePath))
#         rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)


#         bb = align.getLargestFaceBoundingBox(rgbImg)
#         if bb is None:
#             raise Exception("Unable to find a face: {}".format(imagePath))

#         alignedFace = align.align(imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
#         if alignedFace is None:
#             raise Exception("Unable to align image: {}".format(imagePath))

#         rep = net.forward(alignedFace)

#         return rep

#     def classify(self, imagePath):
#         representation = self._represent(imagePath)
#         name = self.classifier.predict([representation])

#         return name

# def test():
#     rec = FaceRecognizer()
#     rec.add_sample('adams', os.path.expanduser("~/openface/images/examples/adams.jpg"))
#     rec.add_sample('clapton', os.path.expanduser("~/openface/images/examples/clapton-1.jpg"))
#     rec.add_sample('clapton', os.path.expanduser("~/openface/images/examples/clapton-2.jpg"))
#     rec.add_sample('lennon', os.path.expanduser("~/openface/images/examples/lennon-1.jpg"))
#     rec.add_sample('lennon', os.path.expanduser("~/openface/images/examples/lennon-2.jpg"))

#     rec.cluster()

#     name = rec.classify(os.path.expanduser("~/openface/images/examples/lennon-2.jpg"))
#     print name[0]
#     assert name == "lennon"

# if __name__ == "__main__":
#     test()