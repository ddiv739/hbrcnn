import tensorflow as tf 
import numpy as np 
import pandas as pd 
import cv2 
import os 
import time

class HBRCNN():

    def __init__(self,model_dir,class_labels=['circle','rectangle','triangle']):
        self.model = tf.keras.models.load_model(model_dir)
        self.required_in_dims = self.model.layers[0].input.shape[1], self.model.layers[0].input.shape[2]
        self.class_labels = class_labels

        self.rescaler = tf.keras.Sequential([
                            tf.keras.layers.experimental.preprocessing.Resizing(self.required_in_dims[0],self.required_in_dims[1]),
                            tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)
                        ])
    
    def draw_annotate(self,img,x,y,w,h,preds):
        pts = np.array([
            (x,y),#TL
            (x+w,y),#TR
            (x+w,y+h),#BR
            (x,y+h),#BL
        ],np.int32)
        cv2.polylines(img,[pts],True,(0,255,0))
        
        preds = preds.tolist()
        max_value = max(preds)
        max_index = preds.index(max_value)

        plabel = f'{self.class_labels[max_index]} : {max_value*100:.2f}'
        print(plabel)
        cv2.putText(img,
            plabel, 
            (x,y-10), #BL xy
            cv2.FONT_HERSHEY_SIMPLEX, #Font
            0.4, #scale
            (0,255,0), #width
            1) #thickness

    def predict(self,input_fp):
        
        if type(input_fp) != str:
            raise Exception('Sorry, one image filepath at a time for this implementation. Maybe next time! :) ') 
        
        start = time.time()
        img = cv2.imread(input_fp)

        #Do a colour correction
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Selective search step
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast(sigma=0.01)
        rects = ss.process()

        print(f'SS found a total of {len(rects)} inputs to be searched!')

        for x,y,w,h in rects:
            if w >= img.shape[0]*0.95 and h>= img.shape[1]*0.95:
                print('Discarding rect pred... Entire img predicted...')
                continue

            roi = img[y:y+h,x:x+w,:]
            roi = self.rescaler(roi)

            preds = self.model.predict(np.expand_dims(roi,axis=0), batch_size =1)

            print(f'x:{x},y:{y}.Predicting: {preds[0]}')
            # max_value = max(preds[0])
            # print(max_value)
            # if max_value < 0.96:
            #     print('discarding ROI, max pred too low')
            #     continue
            self.draw_annotate(img,x,y,w,h,preds[0])

        print(f'Total processing time: {time.time()-start}s or eq: {1/(time.time()-start)} FPS')
        cv2.imshow('asdf',img)
        cv2.waitKey()


if __name__ == '__main__':
    x = HBRCNN('./SavedModels/first_pass')