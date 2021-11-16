import os
import cv2
import numpy as np

def mouse_listener_deer(event, x, y, flags, param):
   global deerPoints
   if event == cv2.EVENT_LBUTTONDOWN:
       deerPoints.append(np.array([x,y]))

def mouse_listener_lion(event, x, y, flags, param):
   global lionPoints
   if event == cv2.EVENT_LBUTTONDOWN:
       lionPoints.append(np.array([x,y]))




deerPoints = list()
lionPoints = list()
deer = cv2.imread( os.getcwd()+'\\'+'dotted-deer.jpg',cv2.IMREAD_UNCHANGED)
lion = cv2.imread( os.getcwd()+'\\'+'dotted-lion.jpg',cv2.IMREAD_UNCHANGED)

lion_file = open("lion-points-test.txt", "w")
deer_file = open("deer-points-test.txt", "w") 

cv2.namedWindow("deer_win")
cv2.setMouseCallback("deer_win", mouse_listener_deer)
cv2.imshow("deer_win",deer)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow("lion_win")
cv2.setMouseCallback("lion_win", mouse_listener_lion)
cv2.imshow("lion_win",lion)
cv2.waitKey(0)
cv2.destroyAllWindows()

deer_file.write(str(len(deerPoints))+'\n')
for point in deerPoints:
    deer_file.write(str(point[0])+' '+str(point[1])+'\n')
deer_file.close()

lion_file.write(str(len(lionPoints))+'\n')
for point in lionPoints:
    lion_file.write(str(point[0])+' '+str(point[1])+'\n')
lion_file.close()