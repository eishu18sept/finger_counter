import cv2
import numpy as np
from sklearn.metrics import pairwise

background = None
acc_wgt = 0.5
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def calc_accum_avg(frame,acc_wgt):
    global background
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,acc_wgt)

def segment(frame,threshold=25):
    diff = cv2.absdiff(background.astype('uint8'),frame)
    ret,thresholded = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    else:
        hand_segment = max(contours,key=cv2.contourArea)
        
        return(thresholded,hand_segment)

#Counting fingers using convex hull

def count_fingers(thresholded,hand_segment):
    conv_hull = cv2.convexHull(hand_segment)
    
    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
    
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    dist = pairwise.euclidean_distances([cX,cY],Y=[left,right,top,bottom])[0]
    
    max_dist = dist.max()
    
    radius = int(0.9*max_dist)
    circumference = 2*3.14*radius
    
    circle_roi = np.zeros(thresholded[:2],dtype='uint8')
    
    cv2.circle(circle_roi,(cX,cY),radius,255,10)
    
    circle_roi = cv2.bitwise_and(thresholded,thresholded,mask=circle_roi)
    
    contours,hierarchy = cv2.findContours(circle_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    count = 0
    
    for cnt in contours:
        
        (x,y,w,h) = cv2.boundingRect(cnt)
        out_of_wrist = (cY + (cY*0.25)) > (y+h)
        limit_points = ((circumference*0.25) > cnt.shape[0])
        
        if out_of_wrist and limit_points:
            count += 1
            
    return count 
    
cam = cv2.VideoCapture(0)

num_frames = 0

while True:
    
    ret,frame = cam.read()
    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
    
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    if num_frames<60:
        calc_accum_avg(gray,acc_wgt)
        
        if num_frames <= 59:
            cv2.putText(frame_copy,"WAIT , GETTING BACKGROUND ",(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger Count',frame_copy)
            
    else:
        
        hand = segment(gray)
        
        if hand is not None:
            thresholded,hand_segment = hand
            
            #draws contour around real hand in live stream
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)
            
            fingers = count_fingers(thresholded,hand_segment)
            
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
            cv2.imshow('Thresholded',thresholded)
            
    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)
    
    num_frames += 1
    
    cv2.imshow("Finger Count",frame_copy)
    
    k = cv2.waitKey(1) & 0xFF 
    if k==27:
        break 
        
cam.release()
cv2.destroyAllWindows()
        
        
