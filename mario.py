import cv2
import pygame, sys
from pygame.locals import *
import numpy as np

def nothing(x):
    pass

hue=179
sat=20
va=140
er=7
backImage=None
min=5
max=100
#cv2.namedWindow('trackbars')
#cv2.createTrackbar('H','trackbars',5,120,nothing)
#cv2.createTrackbar('S','trackbars',100,120,nothing)
# cv2.createTrackbar('H','trackbars',0,179,nothing)
# cv2.createTrackbar('S','trackbars',0,255,nothing)
# cv2.createTrackbar('V','trackbars',0,255,nothing)
# cv2.createTrackbar('e','trackbars',5,10,nothing)
count=0
def cvimage_to_pygame(backImage):
#  print backImage.shape[:2]
  return pygame.image.frombuffer(backImage.tostring(), backImage.shape[1::-1],
                                   "RGB")

def imageIN():
      global backImage
      global table
      global count
      count+=1
      image="images/sequence/"+str(count)+".png"
      img = cv2.imread(image) #original image
      img_backup=cv2.imread(image)

  ######################################################
  #-------------------------------CASCADE
      #carCascade=cv2.CascadeClassifier('data/cas4.xml')
      #carCascade2=cv2.CascadeClassifier('data/cas3.xml')
      grey=cv2.cvtColor(img_backup,cv2.COLOR_BGR2GRAY)
      cv2.equalizeHist(grey)
      #cars = carCascade.detectMultiScale(grey,1.3,5,0,(100,100),(0,0))
      #print len(cars)
      # for (x,y,w,h) in cars:
      #   cv2.rectangle(img_backup,(x,y),(x+w,y+h),(255,0,0),2)
      #   roi_gray = grey[y:y+h, x:x+w]
      #   roi_color = img_backup[y:y+h, x:x+w]
      #   car2 = carCascade2.detectMultiScale(roi_gray)
      #   for (ex,ey,ew,eh) in car2:
      #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

  ################### trackbars#################################
      #hsv=cv2.getTrackbarPos('H','trackbars')
      #sat=cv2.getTrackbarPos('S','trackbars')
      # v=cv2.getTrackbarPos('V','trackbars')
      # er=cv2.getTrackbarPos('e','trackbars')
  ##################################3#################
      #display original
      cv2.namedWindow('original',cv2.WINDOW_NORMAL)
      cv2.imshow('original',img)
  ###############################
      grey=cv2.GaussianBlur(grey,(5,5),0)
      edges=cv2.Canny(grey,min,max) # canny edges

      # upper_blue = np.array([[hue],[sat],[va]])
      # lower_blue = np.array([0,0,0])

      # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #hsv
      #
      # mask = cv2.inRange(hsv, lower_blue, upper_blue)
      # kernel = np.ones((er,er),np.uint8)
      # erosion = cv2.erode(mask,kernel,iterations = 1)
      # kernel = np.ones((er,er),np.uint8)
      # dilation = cv2.dilate(erosion,kernel,iterations = 1)
      backImage=img_backup
      group=np.transpose(edges)
      table=np.zeros((703,1855))
      i=0
      j=0
      for row in group:
        temp=0
        j=0
        for point in row:
          #print len(row)
          if temp==0:
            if point !=0:
              table[j][i]=255
              temp+=1
          j+=1
        i+=1
      #table=np.transpose(table)
      #res = cv2.bitwise_and(img,img, mask= mask)

      # cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
      # cv2.imshow('mask',dilation)
      cv2.namedWindow('edges',cv2.WINDOW_AUTOSIZE)
      cv2.imshow('edges',table)
      cv2.waitKey(4)


rects=[]
data=[]

def getImage():
  imageIN()
  return cvimage_to_pygame(backImage)

def getRects():
  global data
  for i in range(len(table)):
    for j in range(len(table[i])):
      if table[i][j]!=0:
        data.append(i)


imageIN()
#cv2.destroyAllWindows()

pygame.init()
width,height=1855,703
screen=pygame.display.set_mode((width,height))
keys = [False, False, False, False]
playerpos=[700,140]
player = pygame.image.load("player/mario2.png")
coin=pygame.image.load("images/Coin.png")
background=cvimage_to_pygame(backImage)
#background=pygame.image.load("images/streetview.jpeg")

i=0
j=0
getRects()
while 1:

    # 5 - clear the screen before drawing it again
    screen.fill(0)
    # 6 - draw the screen elements
    for x in range(width/background.get_width()+1):
        for y in range(height/background.get_height()+1):
            screen.blit(background,(0,0))
    screen.blit(player,playerpos)
    # 7 - update the screen
    pygame.display.flip()
    # 8 - loop through the events
    for event in pygame.event.get():
        # check if the event is the X button
        if event.type==pygame.QUIT:
            # if it is quit the game
            pygame.quit()
            exit(0)
        if event.type == pygame.KEYDOWN:
            if event.key==K_w:
                keys[0]=True
            elif event.key==K_a:
                keys[1]=True
            elif event.key==K_s:
                keys[2]=True
            elif event.key==K_d:
                keys[3]=True
        if event.type == pygame.KEYUP:
            if event.key==pygame.K_w:
                keys[0]=False
            elif event.key==pygame.K_a:
                keys[1]=False
            elif event.key==pygame.K_s:
                keys[2]=False
            elif event.key==pygame.K_d:
                keys[3]=False
        if playerpos[0]<500:
            playerpos[0]=playerpos[0]+150
            background=getImage()
            getRects()


    if keys[0]:
        playerpos[1]-=10
    elif keys[2]:
        playerpos[1]+=10
    if keys[1]:
        playerpos[0]-=10
    elif keys[3]:
        playerpos[0]+=10
    #print len(table[playerpos[1]])
    if table[ playerpos[1] ][ playerpos[0] ]!=1:
            #print data[playerpos[0]]
            playerpos[1]=data[playerpos[0]]






#cv2.destroyAllWindows()
