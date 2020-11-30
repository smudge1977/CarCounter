import os
import re
import cv2 # opencv library
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
import time

# https://medium.com/machine-learning-world/tutorial-making-road-traffic-counting-app-based-on-computer-vision-and-opencv-166937911660

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# #os.environ["PAFY_BACKEND"] = "internal"

# url = "https://youtu.be/hkMPESRDoTM"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")

# vcap = cv2.VideoCapture()
# vcap.open(best.url)



print(cv2.getBuildInformation())

#fgbg = cv2.createBackgroundSubtractorMOG2()  
fgbg = cv2.createBackgroundSubtractorKNN()
#fgbg = cv2.createBackgroundSubtractorGMG()
font = cv2.FONT_HERSHEY_SIMPLEX



AOI = {"top_left":(50,0),"bottom_right":(700,720)}
GRAY = cv2.COLOR_BGR2GRAY  # How to GRAY scale
THRESH = 50      # Threshold for doing the diff
DILATION = 12    # Dilation kernal 'square' to make a single blog

SHOW = {'THIS':True,'THRESHOLD':True}

def process():
        
    #cap = cv2.VideoCapture("rtsp://Admin:a2345678@10.26.4.121:554", cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture("footage\\Cars2.mp4")

    # cap.set(3,640) # Setting 3 Width
    # cap.set(4,480) # Setting 4 Height
    cap.set(10,100) #Setting 10  Brightness

    ret, this_frame = cap.read()
    # cv2.imshow('This',this_frame)
    this_frame = this_frame[AOI['top_left'][1]:AOI['bottom_right'][1],AOI['top_left'][0]:AOI['bottom_right'][0]]  # y then x !!  y1,y2  x1,x2
    gray_frame = cv2.cvtColor(this_frame, GRAY)

    while(1):
        last_frame = gray_frame
        start_time = time.time()
        ret, this_frame = cap.read()
        
        #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #plt.title("frame: ")
        #plt.show()
        #cv2.imshow('Input',frame)
        
        this_frame = this_frame[AOI['top_left'][1]:AOI['bottom_right'][1],AOI['top_left'][0]:AOI['bottom_right'][0]]  # y then x !!  y1,y2  x1,x2
        gray_frame = cv2.cvtColor(this_frame, GRAY)
        

        diff_frame = cv2.absdiff(gray_frame, last_frame)
        cv2.imshow('Diff',diff_frame)

        # perform image thresholding
        ret, thresh_frame = cv2.threshold(diff_frame, THRESH, 255, cv2.THRESH_BINARY)
        # apply image dilation
        kernel = np.ones((DILATION,DILATION),np.uint8)
        dilated = cv2.dilate(thresh_frame,kernel,iterations = 1)

        cv2.line(dilated, (0, 200),(900,200),(255, 255, 255))
        cv2.imshow('Threshold and dilation',dilated)

        # plt.imshow(dilated)
        # #plt.title("frame: ")
        # plt.show()
        contours, hierarchy = cv2.findContours(dilated.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        valid_cntrs = []
        
        for i,cntr in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cntr)
            valid_cntrs.append(cntr)
            # if (x <= 900) & (y >= 200) & (cv2.contourArea(cntr) >= 25):
            #     valid_cntrs.append(cntr)
            

        # count of discovered contours        
        #len(valid_cntrs)

        dmy = this_frame.copy()
        cv2.putText(dmy, "vehicles detected: " + str(len(valid_cntrs)), (55, 15), font, 0.6, (0, 180, 0), 2)
        cv2.drawContours(dmy, valid_cntrs, -1, (255,200,0), 2)
        cv2.line(dmy, (0, 200),(900,200),(100, 0, 0))
        cv2.imshow('dmy',dmy)


        

        #cv2.imshow('VIDEO', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        print('Took ',(start_time - time.time()))

if __name__ == "__main__":

    process()
    pass




    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    
#     #grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)
# # width, height = 200,300
# # pts1 = np.float32([[255,8],[71,198],[357,489],[299,550]]) # select the input area
# # pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]]) # output area
# # matrix = cv2.getPerspectiveTransform(pts1,pts2)
# # imgOutput = cv2.warpPerspective(img,matrix,(width,height))

#     # setup detectors
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
#     fgmask = fgbg.apply(frame)

#     #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     frame = cv2.Canny(frame,150,350)

# # cv2.imshow('Gray Image',imgGray)

# # cv2.imshow('Canny2',imgCanny2)
# # cv2.imshow('Cropped',imgCropped)


#     cv2.imshow('VIDEO', frame)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()