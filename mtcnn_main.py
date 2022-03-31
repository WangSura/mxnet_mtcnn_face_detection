# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import glob
#sura_修改文件名由main.py到mtcnn_main.py
#sura_直接根据之后的videotest改代码
#sura_检测算法有问题不行有两张脸可能是resize的原因,对头，resize的大小是准确性和美观度的一个考量
detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

#videoinpath  = '/data/wangyue/ssdg_mtcnn_dataset_preprocess/1_1_21_1 copy.avi'
videooutpath = '/data/wangyue/sura_original_code/mxnet_mtcnn_face_detection/chips/'

#videoinpath_list = glob.glob(dataset_path + '**/*.png', recursive=True)#sura_获取指定目录下的所有图片，可是我没有图片？

videoinpath_list = glob.glob('/data/wangyue/ssdg_mtcnn_dataset_preprocess/*.mp4', recursive=True)#sura_获取指定目录下的所有图片，可是我没有图片？
videoinpath_list.sort()
for i in range(len(videoinpath_list)):
    camera = cv2.VideoCapture(videoinpath_list[i])#sura_video路径数组
    while True:
        grab, frame = camera.read()
        #if frame is not None:#sura_报错不能为空但是这是出口因为视频完了
        #img = cv2.resize(frame, (320,320))#sura_图像缩放？？？？但是缩放很丑
        if not grab:break#sura_解决了死循环的问题
        #img=frame#sura_这样不丑但是死循环
        img = frame#sura_暂时要好看
        t1 = time.time()
        results = detector.detect_face(img)
        print('time: ',time.time() - t1)#sura_加括号

        if results is None:
            continue

        total_boxes = results[0]
        points = results[1]

        # extract aligned face chips
        chips = detector.extract_image_chips(img, points, 256, 0.37)#sura_由测试结果图片分析144*144就是这里我们改成256*256
        for j, chip in enumerate(chips):
            #cv2.imshow('chip_'+str(i), chip)#sura_据说是这个bug，没什么大用的代码注释掉
            cv2.imwrite(os.path.join(videooutpath, str(i) + 'chip_'+str(j)+'.png'), chip)#sura_添加路径
            #cv2.imwrite('chip_'+str(i)+'.png', chip)
'''sura_生成图片副本没用
    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (int(p[i]), int(p[i + 5])), 1, (0, 0, 255), 2)#sura_报错又是浮点数转换为整数

    #cv2.imshow("detection result", draw)#sura_据说是这个bug，没什么大用的代码注释掉
    #cv2.waitKey(0)
'''
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
