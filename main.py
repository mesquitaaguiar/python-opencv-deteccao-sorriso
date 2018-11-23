#!/usr/bin/env python

from __future__ import print_function

import cv2 as cv

import video as v

if __name__ == '__main__':
    import sys
    import getopt

    args, sources = getopt.getopt(sys.argv[1:], '', 'shotdir=')
    args = dict(args)
    shotdir = args.get('--shotdir', '.')
    if len(sources) == 0:
        sources = [ 0 ]
    
    faceCascade = cv.CascadeClassifier("lbpcascade_frontalface.xml")
    smileCascade = cv.CascadeClassifier("haarcascade_smile.xml")

    caps = list(map(v.create_capture, sources))
    shot_idx = 0
    while True:
        for i, cap in enumerate(caps):
            ret, img = cap.read()
            
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor= 1.1,
                    minNeighbors=8,
                    minSize=(55, 55),
                    flags=cv.CASCADE_SCALE_IMAGE
                )
            
            sorri = "Apareca, por favor!"
            x , y = 150, 100;
            if(len(faces) > 0 ):
            
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                        
                smile = smileCascade.detectMultiScale(
                                    roi_gray,
                                    scaleFactor= 1.16,
                                    minNeighbors=35,
                                    minSize=(25, 25),
                                    flags=cv.CASCADE_SCALE_IMAGE
                                )
                
                sorri = "Sorria, por favor!"
                for (x2, y2, w2, h2) in smile:
                    cv.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
                    sorri = "Obrigado pelo sorriso!"
                        
            cv.putText(img,sorri,(x,y-7), 1, 1.2, (0, 255, 0), 2, cv.LINE_AA)
            
            cv.imshow('capture %d' % i, img)
        ch = cv.waitKey(1)
        
        if ch == 27:
            break
            
    cv.destroyAllWindows()