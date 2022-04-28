import numpy as np
import cv2
import time
import copy
import pickle
import os, sys
from matplotlib import pyplot as plt
from PIL import Image
import scipy.misc

'''
This file has code needed for running seq_nms after standard yolov5 sends predictions. 
'''

# CLASSES=("__background__","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush")
CLASSES= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
CONF_THRESH = 0.25 # 0.5
NMS_THRESH = 0.3
IOU_THRESH = 0.6
IOU_THRESH_DELETE = 0.3


'''
Takes res as input. Which is [r] where r is [[meta], [scores], [boxes]] | [x] where x is [meta, scores, boxes]
res is a list of r | r is a list of x | x is list of meta, scores, boxes
res is for frames | r is boxes,scores,meta / x for 1 frame | x is for 1 box
res is for all frames
r is for all boxes
x is for a box

which is basically... for each frame, what is its meta and what are scores of each boxes
def createInputs(res):
    create_begin=time.time()
    dets=[[] for i in CLASSES[1:]] # ...
    for cls_ind,cls in enumerate(CLASSES[1:]): # for all classes 
        for frame_ind,frame in enumerate(res): # for all frames in the video  
                    # contains list of items for 1 frame where 1 item is list of scores for each box etc 
                    cls_boxes = np.zeros((len(res[frame_ind]), 4), dtype=np.float64) # for 1 frame
                    cls_scores = np.zeros((len(res[frame_ind]), 1), dtype=np.float64)
                    for box_ind, box in enumerate(frame): # for all items in the frame 
                        # item/box -> meta, score, box | box -> a,b,c,d 
                        cls_boxes[box_ind][0] = box[2][0]-box[2][2]/2
                        cls_boxes[box_ind][1] = box[2][1]-box[2][3]/2
                        cls_boxes[box_ind][2] = box[2][0]+box[2][2]/2
                        cls_boxes[box_ind][3] = box[2][1]+box[2][3]/2
                        if str(box[0], 'utf-8')==cls: # checking the meta/label 
                            cls_scores[box_ind][0] = box[1] # updating the score 
                        else:
                            cls_scores[box_ind][0] = 0.00001
            cls_dets = np.hstack((cls_boxes,cls_scores)).astype(np.float64)
            dets[cls_ind].append(cls_dets)
    create_end=time.time()
    print ('create inputs: {:.4f}s'.format(create_end - create_begin))
    return dets
'''
def createInputs(res):
    create_begin=time.time()
    #print("res----------------")
    #print(res)
    dets=[[] for i in CLASSES[1:]] 
    for cls_ind,cls in enumerate(CLASSES[1:]): # for all classes | cls_ind -> class index | cls-> class 
        for frame_ind,frame in enumerate(res): # for all frames in the video 
            #cls_boxes = np.zeros((len(res[frame_ind]), 4), dtype=np.float64) # lenght of the frame | 4 means a b c d
            #cls_scores = np.zeros((len(res[frame_ind]), 1), dtype=np.float64) # 1 -> main score
            # print("1 frame")
            # print(frame, "\n lenght of frame", len(frame))
            # cls_boxes = np.zeros((len(frame[0][1]), 4), dtype=np.float64) # lenght of the frame | 4 means a b c d
            cls_boxes = np.zeros((len(frame), 4), dtype=np.float64) # lenght of the frame | 4 means a b c d
            cls_scores = np.zeros((len(frame), 1), dtype=np.float64) # 1 -> main score 
            #print("frame[0]:---------------")
            #print(frame[0])
            #print(len(frame[0])) # extra encapsualtion, so frame[1][0]
            # frame=frame[0]
            # print("shape of cls_box:")
            # print(cls_boxes.shape)

            for i in range(len(frame)): # len of frame 1 will tell the number of boxes | should be frame_id? CHECK
                # frame[2] -> all boxes in that frame 
                # i will handle 1 box
                # WHY frame[2][i][0][0]-frame[2][i][0][2]/2 instead of frame[2][i][0]-frame[2][i][2]/2
                # print(len(frame[1])==len(frame[2]))
                # print(len(frame[1]))
                # print(len(frame[2]))
                # print("INSIDE 1 BOX: ")
                # print(frame[i])
                # print(i)
                #cls_boxes[i][0]
                # cls_boxes[i][0] = frame[2][i][0]-frame[2][i][2]/2 # it hsould be modified x1
                cls_boxes[i][0] = frame[i][2][0]-frame[i][2][2]/2 # it hsould be modified x1
                cls_boxes[i][1] = frame[i][2][1]-frame[i][2][3]/2 # modfied x2
                cls_boxes[i][2] = frame[i][2][0]+frame[i][2][2]/2 # y1
                cls_boxes[i][3] = frame[i][2][1]+frame[i][2][3]/2 # y2

                #print(frame[0][i])
                #if(len(frame[0][i]) > 0): # frame 0 is meta | if it has a meta?
                  # if(cls=="bird"):
                  #   print(frame[0], cls, CLASSES[frame[0][i][0]])
                  #   print(np.array(frame[1][0]).max())
                # print(frame[0][0][i])
                # if CLASSES[frame[0][0][i]]==cls: # why 0 
                # print("label is \n", frame[i][0][0], CLASSES[frame[i][0][0]], cls)
                if CLASSES[frame[i][0][0]]==cls: # why 0 
                    if(cls=="bird"):
                      print("MODEL SA ID YES !!!")
                    # print(CLASSES[frame[i][0][0]])
                    # cls_scores[i][0] = np.array(frame[1][i])#[0]).max() # why 0
                    cls_scores[i][0] = np.array(frame[i][1][0])#[0]).max() # why 0
                else:
                    if(cls=="bird"):
                      print("label is: ", CLASSES[frame[i][0][0]], cls, CLASSES[frame[i][0][0]]==cls)                
                    cls_scores[i][0] = 0.00001

            cls_dets = np.hstack((cls_boxes,cls_scores)).astype(np.float64)
            dets[cls_ind].append(cls_dets)
    create_end=time.time()
    print ('create inputs: {:.4f}s'.format(create_end - create_begin))
    return dets

def createLinks(dets_all):
    link_begin=time.time()
    links_all=[]
    #建立每相邻两帧之间的link关系
    frame_num=len(dets_all[0])
    cls_num=len(CLASSES)-1
    #links_all=[] #保存每一类的全部link，第一维为类数，第二维为帧数-1，为该类下的links即每一帧与后一帧之间的link，第三维每帧的box数，为该帧与后一帧之间的link
    for cls_ind in range(cls_num): #第一层循环，类数
        links_cls=[] #保存一类下全部帧的links
        for frame_ind in range(frame_num-1): #第二层循环，帧数-1，不循环最后一帧
            dets1=dets_all[cls_ind][frame_ind]
            dets2=dets_all[cls_ind][frame_ind+1]
            box1_num=len(dets1)
            box2_num=len(dets2)
            #先计算每个box的area
            if frame_ind==0:
                areas1=np.empty(box1_num)
                for box1_ind,box1 in enumerate(dets1):
                    areas1[box1_ind]=(box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
            else: #当前帧的area1就是前一帧的area2，避免重复计算
                areas1=areas2
            areas2=np.empty(box2_num)
            for box2_ind,box2 in enumerate(dets2):
                areas2[box2_ind]=(box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
            #计算相邻两帧同一类的link
            links_frame=[] #保存相邻两帧的links
            for box1_ind,box1 in enumerate(dets1):
                area1=areas1[box1_ind]
                x1=np.maximum(box1[0],dets2[:,0])
                y1=np.maximum(box1[1],dets2[:,1])
                x2=np.minimum(box1[2],dets2[:,2])
                y2=np.minimum(box1[3],dets2[:,3])
                w =np.maximum(0.0, x2 - x1 + 1)
                h =np.maximum(0.0, y2 - y1 + 1)
                inter = w * h
                ovrs = inter / (area1 + areas2 - inter)
                links_box=[ovr_ind for ovr_ind,ovr in enumerate(ovrs) if ovr >= IOU_THRESH] #保存第一帧的一个box对第二帧全部box的link
                links_frame.append(links_box)
            links_cls.append(links_frame)
        links_all.append(links_cls)
    link_end=time.time()
    print ('link: {:.4f}s'.format(link_end - link_begin))
    return links_all

  
def maxPath(dets_all,links_all):
    max_begin=time.time()
    for cls_ind,links_cls in enumerate(links_all):
        dets_cls=dets_all[cls_ind]
        while True:
            rootindex,maxpath,maxsum=findMaxPath(links_cls,dets_cls)
            if len(maxpath) <= 1:
                break
            rescore(dets_cls,rootindex,maxpath,maxsum)
            deleteLink(dets_cls,links_cls,rootindex,maxpath,IOU_THRESH_DELETE)
    max_end=time.time()
    print ('max path: {:.4f}s'.format(max_end - max_begin))

def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def NMS(dets_all):
    for cls_ind,dets_cls in enumerate(dets_all):
        for frame_ind,dets in enumerate(dets_cls):
            keep=nms(dets, NMS_THRESH)
            dets_all[cls_ind][frame_ind]=dets[keep, :]

def findMaxPath(links,dets):
    maxpaths=[] #保存从每个结点到最后的最大路径与分数
    roots=[] #保存所有的可作为独立路径进行最大路径比较的路径
    maxpaths.append([ (box[4],[ind]) for ind,box in enumerate(dets[-1])])
    for link_ind,link in enumerate(links[::-1]): #每一帧与后一帧的link，为一个list
        curmaxpaths=[]
        linkflags=np.zeros(len(maxpaths[0]),int)
        det_ind=len(links)-link_ind-1
        for ind,linkboxes in enumerate(link): #每一帧中每个box的link，为一个list
            if linkboxes == []:
                curmaxpaths.append((dets[det_ind][ind][4],[ind]))
                continue
            linkflags[linkboxes]=1
            prev_ind=np.argmax([maxpaths[0][linkbox][0] for linkbox in linkboxes])
            prev_score=maxpaths[0][linkboxes[prev_ind]][0]
            prev_path=copy.copy(maxpaths[0][linkboxes[prev_ind]][1])
            prev_path.insert(0,ind)
            curmaxpaths.append((dets[det_ind][ind][4]+prev_score,prev_path))
        root=[maxpaths[0][ind] for ind,flag in enumerate(linkflags) if flag == 0]
        roots.insert(0,root)
        maxpaths.insert(0,curmaxpaths)
    roots.insert(0,maxpaths[0])
    maxscore=0
    maxpath=[]
    for index,paths in enumerate(roots):
        if paths==[]:
            continue
        maxindex=np.argmax([path[0] for path in paths])
        if paths[maxindex][0]>maxscore:
            maxscore=paths[maxindex][0]
            maxpath=paths[maxindex][1]
            rootindex=index
    return rootindex,maxpath,maxscore

def rescore(dets, rootindex, maxpath, maxsum):
    newscore=maxsum/len(maxpath)
    for i,box_ind in enumerate(maxpath):
        dets[rootindex+i][box_ind][4]=newscore

def deleteLink(dets,links, rootindex, maxpath,thesh):
    for i,box_ind in enumerate(maxpath):
        areas=[(box[2]-box[0]+1)*(box[3]-box[1]+1) for box in dets[rootindex+i]]
        area1=areas[box_ind]
        box1=dets[rootindex+i][box_ind]
        x1=np.maximum(box1[0],dets[rootindex+i][:,0])
        y1=np.maximum(box1[1],dets[rootindex+i][:,1])
        x2=np.minimum(box1[2],dets[rootindex+i][:,2])
        y2=np.minimum(box1[3],dets[rootindex+i][:,3])
        w =np.maximum(0.0, x2 - x1 + 1)
        h =np.maximum(0.0, y2 - y1 + 1)
        inter = w * h
        ovrs = inter / (area1 + areas - inter)
        deletes=[ovr_ind for ovr_ind,ovr in enumerate(ovrs) if ovr >= thesh] #保存待删除的box的index
        for delete_ind in deletes:
            if delete_ind!=box_ind:
                dets[rootindex+i][delete_ind, 4] = 0
        if rootindex+i<len(links): #除了最后一帧，置box_ind的box的link为空
            for delete_ind in deletes:
                links[rootindex+i][delete_ind]=[]
        if i > 0 or rootindex>0:
            for priorbox in links[rootindex+i-1]: #将前一帧指向box_ind的link删除
                for delete_ind in deletes:
                    if delete_ind in priorbox:
                        priorbox.remove(delete_ind)





def dsnms(res):
    dets=createInputs(res)
    links=createLinks(dets)
    maxPath(dets,links)
    NMS(dets)
    boxes=[[] for i in dets[0]]
    classes=[[] for i in dets[0]]
    scores=[[] for i in dets[0]]    
    for cls_id, det_cls in enumerate(dets):
        for frame_id, frame in enumerate(det_cls):
            for box_id, box in enumerate(frame):
                # print("ENDGAME")
                # print(box)
                if box[4] >= CONF_THRESH:
                    ymin = box[1]
                    xmin = box[0]
                    ymax = box[3]
                    xmax = box[2]
                    # if box[4] != []:
                    boxes[frame_id].append(np.array([ymin, xmin, ymax, xmax]))
                    classes[frame_id].append(cls_id+1)
                    scores[frame_id].append(box[4])
    return boxes, classes, scores