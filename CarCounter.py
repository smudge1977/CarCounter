#!/usr/env phython3

import cv2
import numpy as np


cap = cv2.VideoCapture("footage\\cars.mp4")
ret, this_frame = cap.read()
cv2.imshow('This1',this_frame)
k = cv2.waitKey(30)