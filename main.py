# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)


img = cv2.imread('test2.jpg')

# run detector
results = detector.detect_face(img)

if results is not None:

    total_boxes = results[0]
    points = results[1]
    
    # extract aligned face chips
    chips = detector.extract_image_chips(img, points, 256, 0.37)#sura_由测试结果图片分析144*144就是这里我们改成256*256
    for i, chip in enumerate(chips):
        #cv2.imshow('chip_'+str(i), chip)#sura_据说是这个bug，没什么大用的代码注释掉
        cv2.imwrite('chip_'+str(i)+'.png', chip)

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (int(p[i]), int(p[i + 5])), 1, (0, 0, 255), 2)#sura_报错又是浮点数转换为整数

    #cv2.imshow("detection result", draw)#sura_据说是这个bug，没什么大用的代码注释掉
    cv2.waitKey(0)

# --------------
# test on camera
# --------------
'''
camera = cv2.VideoCapture(0)
while True:
    grab, frame = camera.read()
    img = cv2.resize(frame, (320,180))

    t1 = time.time()
    results = detector.detect_face(img)
    print 'time: ',time.time() - t1

    if results is None:
        continue

    total_boxes = results[0]
    points = results[1]

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
    #cv2.imshow("detection result", draw)#sura_据说是这个bug，没什么大用的代码注释掉
    cv2.waitKey(30)
'''
