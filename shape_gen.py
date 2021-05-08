import cv2
import numpy as np 
from math import sin,cos,tan,radians
import random 
import os 
import json

def check_points_pos(points,imgdim):
    for point in points:
        if any(x<0 or x>imgdim[0] for x,y in points) or any(y<0 or y>imgdim[1] for x,y in points):
            return False

    return True

def get_rotated_point(c,p, theta):
    # cx, cy - center of square coordinates
    #x, y - coordinates of a corner point of the square
    # theta is the angle of rotation
    cx,cy = c
    x,y = p
    # translate point to origin
    tempX = x - cx
    tempY = y - cy
    theta = radians(theta)
    #now apply rotation
    rotatedX = tempX*cos(theta) - tempY*sin(theta)
    rotatedY = tempX*sin(theta) + tempY*cos(theta)

    #translate back
    x = rotatedX + cx
    y = rotatedY + cy

    return (x,y)

def draw_circle(img,color,center,radius):
    x,y = center
    colors = {
        'blue':(255,0,0),
        'green':(0,255,0),
        'red':(0,0,255)
    }

    if type(color)==str and color in colors:
        cv2.circle(img,center,radius,colors[color.lower()],-1)
    else:
        if type(color)==str:
            raise Exception('choose red,blue or green else declare a tuple of (B,G,R) with vals 0..255')

        cv2.circle(img,center,radius,color,-1)
    #points are purely for bounding box
    pts = np.array([(x-radius,y-radius),#TL
                (x+radius,y-radius),#TR
                (x+radius,y+radius),#BR
                (x-radius,y+radius),#BL
                ], np.int32)

    return pts

def get_rect(centre:tuple,dims:tuple,rotation:float,draw:bool=False,img=None,color=None):
    rotation = round(rotation%2,2)*90

    x,y = centre
    width = dims[0]//2
    height = dims[1]//2
    
    pts = np.array([get_rotated_point((x,y),(x-width,y-height),rotation),#TL
                    get_rotated_point((x,y),(x+width,y-height),rotation),#TR
                    get_rotated_point((x,y),(x+width,y+height),rotation),#BR
                    get_rotated_point((x,y),(x-width,y+height),rotation),#BL
                    ], np.int32)
    if draw:
        cv2.fillPoly(img,[pts],color)
    return pts

def get_triangle(centre,size,rotation,draw=False,img=None,color=None):

    rotation = round(rotation%1,2)*360
    x,y = centre
    
    pts = np.array([get_rotated_point((x,y),(x,y-abs((size/2)/cos(radians(30)))),rotation),#TR
                    get_rotated_point((x,y),(x-(size*cos(radians(60))),y+((size/2)*tan(radians(30)))),rotation),#BR
                    get_rotated_point((x,y),(x+(size*cos(radians(60))),y+(size/2)*tan(radians(30))),rotation),#BL
                    ], np.int32)

    if draw:
        cv2.fillPoly(img,[pts],color)
    return pts

def __draw_obj(img,pts,color,func,close=False):
    colors = {
        'blue':(255,0,0),
        'green':(0,255,0),
        'red':(0,0,255)
    }

    if type(color)==str and color in colors:
        if close:
            func(img,[pts],True,colors[color.lower()])
        else:
            func(img,[pts],colors[color.lower()])
    else:
        if type(color)==str:
            raise Exception('choose red,blue or green else declare a tuple of (B,G,R) with vals 0..255')
        if close:
            func(img,[pts],True,color)
        else:
            func(img,[pts],color)

def draw_shape(img,pts,color):
    __draw_obj(img,pts,color,cv2.fillPoly)

def draw_outline(img,pts,color='green'):
    __draw_obj(img,pts,color,cv2.polylines,close=True)

def get_shape_outline(pts):
    minx=miny=9999999999
    maxx=maxy=0
    for x,y in pts:
        if x<minx:
            minx=x
        if x>maxx:
            maxx=x
        
        if y<miny:
            miny=y
        if y>maxy:
            maxy=y

    
    return np.array([(minx,miny),#TL
                    (maxx,miny),#TR
                    (maxx,maxy),#BR
                    (minx,maxy),#BL
                    ], np.int32)
    
def get_training_example(width,height):

    blanknp = np.zeros((height,width,3), np.uint8)
    

    img = blanknp

    predef_colors = [(random.randint(5,255),random.randint(5,255),random.randint(5,255)) for i in range(3)]
    
    # cv2.rectangle(img,(50,50),(245,234),(0,255,0),-1)
    bbox = []
    i=0
    while not i:
        if (random.randint(0,1) == 1):
            pts = [[-1,-1]]
            while(not check_points_pos(pts,(width,height))):
                pts = get_rect((width*random.random(),height/1.35*random.random()),(width*0.8*(random.randint(3,10)/10),height*0.8*(random.randint(3,10)/10)),random.random())
            draw_shape(img,pts,predef_colors[i])
            i+=1
            bbox.append(('rectangle',get_shape_outline(pts)))
        # draw_outline(img,pts,'green')
        if (random.randint(0,1) == 1):
            pts = [[-1,-1]]
            while(not check_points_pos(pts,(width,height))):
                pts = get_triangle((width*random.random(),height*random.random()),width/2*(random.randint(3,10)/10),random.random())
            cv2.fillPoly(img,[pts],predef_colors[i])
            i+=1
            bbox.append(('triangle',get_shape_outline(pts)))
        # draw_outline(img,pts,'green')
        if (random.randint(0,1) == 1):
            pts = [[-1,-1]]
            radius = int(width/4*(random.randint(4,11)/10))
            pts = draw_circle(img,predef_colors[i],(random.randint(radius,width-radius),random.randint(radius,height-radius)),radius)
            i+=1
            bbox.append(('circle',get_shape_outline(pts)))

    return img,bbox

def generate_classifier_training_sets(folder_name,num_samples,size):

    os.mkdir(folder_name)
    labelset = {'triangle':0,'circle':0,'rectangle':0}
    for label in ['triangle','circle','rectangle']:
        os.mkdir(folder_name+'/'+label)

    for i in range(num_samples):
        img,bbox = get_training_example(size[0],size[1])

        for label,(tl,tr,bl,br) in bbox:
            x,y = tl
            w = tr[0]-tl[0]
            h = bl[1] - tl[1]
            roiimg = img[tl[1]:br[1],tl[0]:tr[0],:]

            cv2.imwrite(f'{os.getcwd()}/{folder_name}/{label}/{labelset[label]}.png',roiimg)
            labelset[label]+=1

def generate_object_det_sets(folder_name,num_samples,size,save_bboxes=False):
    os.mkdir(folder_name)
    os.mkdir(folder_name+'/images')
    os.mkdir(folder_name+'/annotations')

    for i in range(num_samples):
        img,bboxes = get_training_example(size[0],size[1])

        cv2.imwrite(f'{os.getcwd()}/{folder_name}/images/{i}.png',img)

        if save_bboxes:
            for j in range(len(bboxes)):
                bboxes[j]= (bboxes[j][0],bboxes[j][1].tolist())
            json.dump(bboxes,open(f'{os.getcwd()}/{folder_name}/annotations/{i}.json','w+'))

def show_example_selectsearch(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast(sigma=0.01)
    rects = ss.process()
    print(len(rects))
    for x,y,w,h in rects:
        draw_outline(img,np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)]))
    cv2.imshow('example_outlines_ss_fast',img)
    cv2.waitKey() 

# generate_classifier_training_sets('shape_classifier_data',5000,(360,360))

generate_object_det_sets('classification_data',1000,(360,360),save_bboxes=True)
# img,bbox = get_training_example(360,360)

# show_example_selectsearch(img)
# for i in bbox:
#     draw_outline(img,i[1])

# cv2.imshow('smoigle',img)
# cv2.waitKey() 

