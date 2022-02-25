# import librairies that we use
import numpy as np
import cv2

################################################################################
# THRESHOLDING FUNCTION IMPLEMENTATION
def thresholding(img):
    # visualizing image in HSV parameters
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # the values for lowerWhite and upperWhite are found by tweaking the HSV min/max params in the 
    # trackbar by running ColorPickerScript.py
    lowerWhite = np.array([80, 0, 0])
    upperWhite = np.array([255, 160, 255])
    # passing the values of lowerWhite and upperWhite to create the mask
    maskWhite = cv2.inRange(imgHSV, lowerWhite, upperWhite)
    return maskWhite

list_of_curve=[]
val=10

#this functio we give to it img and we return a curve of img
def getCurve(image,display=2): 
    ## get values of heigth and width
    heigth,width,_=image.shape
    pts=valTrackbars()
    img_warp=warp_img(thresholding(image), pts, width , heigth )#warp_img(image,pts,width,heigth)
    pt_warp=points(image.copy(),pts)
    imgResult=image.copy()
    ## daraja dyal dora
    base,hist_of_img=data_pixel(image,display=True,mini=0.5,r=4)
    midean,hist_of_img=data_pixel(image,display=True,mini=0.9)
    average_of_curve=base-midean
    ## ina jiha ghaydor
    list_of_curve.append(average_of_curve)
    if len(list_of_curve)>val:
        list_of_curve.pop(0)
    curve=int(sum(list_of_curve)/len(list_of_curve))
    
    # display this shit
    if display != 0:
        imgInvWarp = warp_img(img_warp, pts, width, heigth, inverse=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:heigth // 3, 0:width] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (width // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (width // 2, midY), (width // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((width // 2 + (curve * 3)), midY - 25), (width // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = width // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        #cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        # stack all the windows together (just for display purposes, no other requirement)
        imgStacked = stackImages(0.7, ([img, pt_warp, img_warp],
                                             [hist_of_img, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Result', imgResult)
 
    ### make curve between values 1 and -1
    curve = curve/100
    if curve>1: curve = 1
    if curve<-1: curve = -1
 
    return curve
    
    # cv2.imshow("thres",thres_img)
    # cv2.imshow("wrap",img_warp)
    # cv2.imshow("histogramme",hist_of_img)
    


# def limit_of_road(img):
#     imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)  #the convert img to black and white
#     lower,uper=np.array([10,0,0]),np.array([100,255,255])
#     return cv2.inRange(imgHSV,lower,uper)

# this function change curve of img 
def warp_img(img,pts,width,heigth,inverse=False):
    # for define our points
    pt1= np.array(pts,dtype='f')
    pt2=np.array([[0,0],[width,0],[0,heigth],[width,heigth]],dtype='f')
    
    #for convert matrix from img1 to img2
    if inverse:
        mt=cv2.getPerspectiveTransform(pt2,pt1)
    else:
        mt=cv2.getPerspectiveTransform(pt1,pt2)
        
    return cv2.warpPerspective(img,mt,(width,heigth))

def walo(action):
    pass
# create trackbars 
def initializeTrackbars(initialTrackbarVals, wT=480, hT=240):
    # create trackbar window
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initialTrackbarVals[0], wT//2, walo)
    cv2.createTrackbar("Height Top", "Trackbars", initialTrackbarVals[1], hT, walo)
    cv2.createTrackbar("Width Bottom", "Trackbars", initialTrackbarVals[2], wT//2, walo)
    cv2.createTrackbar("Height Bottom", "Trackbars", initialTrackbarVals[3], hT, walo)

# find the value of trackbars (real-time) for my module
def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    # return the bounding coordinates
    points = np.array([(widthTop, heightTop), (wT-widthTop, heightTop), (widthBottom, heightBottom), (wT-widthBottom, heightBottom)],dtype='f')
    return points

def points(img,pt):
    for i in range(4):
        cv2.circle(img,(int(pt[i][0]), int(pt[i][1])),12 ,(0,0,255), cv2.FILLED)
    return img

# make rebot able to bach idor
def data_pixel(tswira,display=False,mini=0.1,r=1):
    if r==1:
        sum_pixel= np.sum(tswira,axis=0)
    else:
        sum_pixel= np.sum(tswira[tswira.shape[0]//r:,:],axis=0)
    path=np.max(sum_pixel)
    noise_path=mini*path
    indx_of_value=np.where(sum_pixel>=noise_path)
    base=int( np.average(indx_of_value) )
    # draw hitogaramme
    if display:
        hist=np.zeros((tswira.shape[0],tswira.shape[1],3),np.uint8)
        for x,i in enumerate(sum_pixel):
            # print("bbb",(x,tswira.shape[0]))
            # cv2.line(hist,(x,tswira.shape[0]),(x,i//255),(0,120,255),1)
            cv2.circle(hist,(base,tswira.shape[0]),20,(0,255,255),cv2.FILLED)
        return base,hist
    return base
        
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
        
if __name__=="__main__": 
    nb=0
    video=cv2.VideoCapture("road.mp4")# for run video
    trackbar=[102, 80, 20, 214]
    initializeTrackbars(trackbar)
    while True:
        # for knew wich frame we are playing with
        nb+=1
        if video.get(cv2.CAP_PROP_FRAME_COUNT) == nb:
            video.set(cv2.CAP_PROP_POS_FRAMES,0)
            nb=0
        
        valid,img=video.read()
        img=cv2.resize(img,(480, 240),interpolation = cv2.INTER_AREA)
        # curve = getLaneCurve(img, display=2)
        # # print the stack of images
        # print(curve)
        curve_result=getCurve(img)
        print(curve_result)
        cv2.imshow("Video for detection",img)
        cv2.waitKey(1)