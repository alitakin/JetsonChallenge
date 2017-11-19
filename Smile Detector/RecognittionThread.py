# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:32:31 2017

@author: Khazar
"""
import threading
import Live_smile_detector as lsd

class RecognitionThread(threading.Thread):

   def __init__(self):
       threading.Thread.__init__(self)
       while(True):
           lsd.Live_smile_detector.getFrame()
           lsd.Live_smile_detector.setSmile()