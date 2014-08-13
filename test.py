import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()


# import cv2
# import numpy as np
# img = cv2.imread('test1.jpg',0)
# img = cv2.medianBlur(img,5)
# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
#
# circles = np.uint16(np.around(circles))
# # for i in circles[0,:]:
# #     # draw the outer circle
# #     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
# #     # draw the center of the circle
# #     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
# #
# cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)
# cv2.imshow('detected circles',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # gray=np.float32(gray)
# # dst =cv2.cornerHarris(gray,2,3,0,04)
# # dst = cv2.dilate(dst,None)
# #
# # img[dst>0.01*dst.max()]=[0,0,255]
# # #cv2.namedWindow("main", cv2.CV_WINDOW_AUTOSIZE)
# # cv2.namedWindow('main',cv2.WINDOW_NORMAL)
# # cv2.imshow('main',dst)
# # if cv2.waitKey(0) & 0xff == 27:
# #         cv2.destroyAllWindows()
# #
# #
#
#
#
#
#
#
#
#
#
# # cap=cv2.VideoCapture(0)
# # while(1):
# #         _,frame=cap.read()
#
# # 	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
# #         lower_blue = np.array([110,50,50])
# #         upper_blue = np.array([130,255,255])
# #         mask = cv2.inRange(hsv, lower_blue, upper_blue)
# #         k = cv2.waitKey(5) & 0xFF #setting a waitkey value
# #         res = cv2.bitwise_and(frame,frame, mask= mask)
# #         cv2.imshow('frame',frame)
# #         cv2.imshow('mask',mask)
# #         cv2.imshow('res',res)
#
# # 	if k==27:
# #                 break
