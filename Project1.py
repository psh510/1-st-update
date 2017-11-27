import cv2
import numpy as np
import sys


mouse_x1,mouse_y1,mouse_x2,mouse_y2=0,0,50,50
mouse_x3,mouse_y3=0,0
cut = False

def draw_mouse(event,x,y,flags,data):
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,mouse_x3,mouse_y3,cut
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x1 = x
        mouse_y1 = y
        if cut == True:
            mouse_x1 += mouse_x3
            mouse_y1 += mouse_x3
        print(x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if cut == True:
            mouse_x3 = x-mouse_x1-mouse_x2
            mouse_y3 = y-mouse_x2-mouse_y2
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_x2 = x
        mouse_y2 = y
        if cut == True:
            mouse_x2 += mouse_x3
            mouse_y2 += mouse_x3
        print(x,y)

def Sub_img(original):
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,mouse_x3,mouse_y3
    sub_img =original[mouse_y1:mouse_y2,mouse_x1:mouse_x2]
    #sub_img = original[mouse_y1+mouse_y3:mouse_y2+mouse_y3,mouse_x1+mouse_x3:mouse_x2+mouse_x3]
    #if cut == True:
     #   sub_img = original[mouse_y1+mouse_y3:mouse_y2+mouse_y3,mouse_x1+mouse_x3:mouse_x2+mouse_x3]
     #   return sub_img
    return sub_img

def imageload():
    global mouse_x1,mouse_y1,mouse_x2,mouse_y2,mouse_x3,mouse_y3,cut
    a = sys.argv[1]
    original = cv2.imread(a,1)
    gray = cv2.imread(a,0)
    #auto = cv2.namedWindow('imgloadview',cv2.WINDOW_NORMAL)
    #img1 = cv2.resize(img, (500, 500)) 
    b=original
    NORMAL = cv2.WINDOW_NORMAL
    AUTO = cv2.WINDOW_AUTOSIZE
    c= NORMAL
    #cv2.namedWindow('imgloadview',c)
    #cv2.setMouseCallback('imgloadview',draw_mouse,param=original)
    while True:
        cv2.namedWindow('imgloadview',c)
        cv2.setMouseCallback('imgloadview',draw_mouse,param=original)
        cv2.imshow('imgloadview',b)
        print(b.shape)
        ori_x,ori_y= b.shape[:2]
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
        elif k == ord('n'):
            c=NORMAL
            cv2.destroyAllWindows()
        elif k == ord('c'):
            d=Sub_img(original)
            b=cv2.resize(d,(ori_x,ori_y))
            cut = True
        elif k == ord('1'):
            b=original
            c=NORMAL
        if cut == True:
            if  k == ord('d'):
                b=Sub_img(original)
                cv2.resize(b,(ori_x,ori_y))


            
             
imageload()