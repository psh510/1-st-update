import cv2
import numpy as np
import sys


def onMOuse(X):
    pass

def imageload():
    a = sys.argv[1]
    original = cv2.imread(a,1)
    gray = cv2.imread(a,0)
    #auto = cv2.namedWindow('imgloadview',cv2.WINDOW_NORMAL)
    #img1 = cv2.resize(img, (500, 500)) 
    b=original
    NORMAL = cv2.WINDOW_NORMAL
    AUTO = cv2.WINDOW_AUTOSIZE
    c= NORMAL
    #cv2.namedWindow('practice2',cv2.WINDOW_AUTOSIZE)
    while True:
        cv2.namedWindow('imgloadview',c)
        cv2.imshow('imgloadview',b)
        print(b.shape)
        k = cv2.waitKey(0)
        if k== 27:
            cv2.destroyAllWindows()
            break
        elif k == ord('g'):
            b=gray
        elif k == ord('r'):
            b=original
        elif k == ord('a'):
            c=AUTO
            cv2.destroyAllWindows()
         #   c=AUTO
        elif k == ord('n'):
            c=NORMAL
            cv2.destroyAllWindows()
            #c=NORMAL
            

 
            
imageload()