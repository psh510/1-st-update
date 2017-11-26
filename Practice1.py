import numpy as np
import cv2


def dot(imgfile1):

    img1 = cv2.imread(imgfile1,1)
    img = cv2.resize(img1, (500, 500)) 
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    print(img[250,250])
    print(b[250, 250], g[250, 250], r[250, 250])

dot('/home/parksanghyeon/Downloads/dot1.png')

def showImage():
    imgfile= '/home/parksanghyeon/Downloads/practice1.png'
   
    img = cv2.imread(imgfile,cv2.IMREAD_COLOR) #second argument means output IMREAD_COLOR=1
                                                #RGB
    img2 = cv2.imread(imgfile,cv2.IMREAD_GRAYSCALE) #second argument means output IMREAD_GRAYSCALE=0
                                                    #GRAY
    #img3 = cv2.imread(imgfile,cv2.IMREAD_UNCHANGED)#second argument means output IMREAD_UNCHANGED=-1
                                                   #RGBA
   
    cv2.namedWindow('practice1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('practice2',cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('practice3',cv2.WINDOW_NORMAL)

    cv2.imshow('cutting',img)
    
    subimg = img[300:400, 350:750]
    cv2.imshow('cutting', subimg)

    img[300:400,0:400] = subimg
    
    cv2.imshow('modified',img)
    
    # b, g, r = cv2.split(img)

    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    print(img[100,100])
    print(b[100, 100], g[100, 100], r[100, 100])
    
    cv2.imshow('blue channel', b)
    cv2.imshow('green channel', g)
    cv2.imshow('red channel',r)

    merged_img = cv2.merge((b,g,r))
    cv2.imshow('merged', merged_img)

    
    #cv2.imshow('practice1', img1)
    #cv2.imshow('practice2', img2)
    #cv2.imshow('practice3', img3)
    print(img.shape)
    print(img2.shape)


    k =cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    if k == ord('c'):
        cv2.imwrite('/home/parksanghyeon/Downloads/practice2.png',img1)
        cv2.destoryAllWindows() #if i type the 'c', it means copy practice1. to pratice3

def onMouse(x):
    pass

def imgBlending(imgfile1, imgfile2):
    im1 = cv2.imread(imgfile1)
    im2 = cv2.imread(imgfile2)
    img1 = cv2.resize(im1, (960, 540)) 
    img2 = cv2.resize(im2, (960, 540)) 

    cv2.namedWindow('ImgPane')
    cv2.createTrackbar('MIXING', 'ImgPane',0,100,onMouse)
    mix = cv2.getTrackbarPos('MIXING', 'ImgPane')

    while True:
        img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100, 0)
        cv2.imshow('ImgPane', img)
        
        k= cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        mix = cv2.getTrackbarPos('MIXING', 'ImgPane')
    cv2.destroyAllWindows()



def addImage(imgfile1, imgfile2):
    
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)
    im1 = cv2.resize(img1, (960, 540)) 
    im2 = cv2.resize(img2, (960, 540)) 
    cv2.imshow('img1',im1)
    cv2.imshow('img2',im2)
    
    add_img1 = im1 + im2
    add_img2 = cv2.add(im1, im2)
    
    cv2.imshow('im1 + im2',add_img1)
    cv2.imshow('add(im1,im2)',add_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#showImage()

#addImage('/home/parksanghyeon/Downloads/practice1.png','/home/parksanghyeon/Downloads/practice1-1.png')
#imgBlending('/home/parksanghyeon/Downloads/practice1.png','/home/parksanghyeon/Downloads/practice1-1.png')