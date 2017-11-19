# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:52:57 2017

@author: Khazar
"""
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import relu6, DepthwiseConv2D

class MainStream:
   'Common base class for all employees'
   empCount = 0
   face

   def __init__(self, name, salary):
      
      MainStream.empCount += 1
   
   def getFace(self):
       
       return self.face
#     print "Total Employee %d" % Employee.empCount

   def setFace(self):
       
#      print "Name : ", self.name,  ", Salary: ", self.salary