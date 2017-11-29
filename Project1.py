import cv2
import numpy as np
import sys
import os

mouse_x1,mouse_y1,mouse_x2,mouse_y2=0,0,0,0
mouse_x3,mouse_y3=0,0
nf=1
cut = False
tran = False
def allfiles(path):
    res = []

    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)

        for file in files:
            filepath = os.path.join(rootpath, file)
            res.append(filepath)

    return res


def draw_mouse(event,x,y,flags,data):
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,mouse_x3,mouse_y3,cut,tran
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x1 = x
        mouse_y1 = y      
        if cut == True:
            tran = True
        print(x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if tran == True:
            mouse_x3 = x
            mouse_y3 = y
    elif event == cv2.EVENT_LBUTTONUP:
        if cut == True:
            tran = False
        mouse_x2 = x
        mouse_y2 = y
        print(x,y)       

def Sub_img(original):
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,mouse_x3,mouse_y3,cut,tran
    if cut==False:
       sub_img =original[mouse_y1:mouse_y2,mouse_x1:mouse_x2]
    if tran==True:
        sub_img =original[mouse_y1+mouse_y3:mouse_y2+mouse_y3,mouse_x1+mouse_x3:mouse_x2+mouse_x3]
    return sub_img


def imageload():
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,mouse_x3,mouse_y3,cut,tran,nf
    out=0
    res = []
    res = allfiles(sys.argv[1])
    ds = len(res)
    while True:
        mouse_x1,mouse_y1,mouse_x2,mouse_y2,mouse_x3,mouse_y3=0,0,0,0,0,0
        a = res[nf]
        original = cv2.imread(a,1)
        gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
        NORMAL = cv2.WINDOW_NORMAL
        AUTO = cv2.WINDOW_AUTOSIZE
        b=original
        c= NORMAL  
        while True:
            cv2.namedWindow('imgloadview',c)
            cv2.setMouseCallback('imgloadview',draw_mouse,param=original)
            cv2.imshow('imgloadview',b)
            ori_x,ori_y= original.shape[:2]
            print(b.shape)
            k = cv2.waitKey(0)
            if k== 27:
                cv2.destroyAllWindows()
                out=1
                break
            elif k == ord('g'):
                b=gray
            elif k == ord('r'):
                b=original
            elif k == ord('a'):
                c=AUTO
                cv2.destroyAllWindows()
            elif k == ord('n'):
                c=NORMAL
                cv2.destroyAllWindows()
            elif k == ord('c'):
                d=Sub_img(original)
                b=cv2.resize(d,(ori_y,ori_x))
                cut =True
            elif k == ord('m'):
                if cut == True:
                    d=Sub_img(original)
                    b=cv2.resize(d,(ori_y,ori_x))   
            elif k == ord('8'):
                if nf < ds-1 :
                    nf = nf+1
                    break
            elif nf == ds-1 :
                nf = 1
                break
            elif k == ord('9'):
                if nf > 1 :
                    nf = nf-1
                    break
            elif nf == 1 :
                nf = ds-1
                break            
    
        if out==1:
            break   
            
imageload()