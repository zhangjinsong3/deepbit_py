# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:05:38 2017

@author: 01376022

图像检索 deepbit 的 python inference

"""

import numpy as np
import re
import sys,os
import os.path
import shutil
#设置当前目录
caffe_root = '/home/zjs/projects/ImageRetrieval/deepbit'
projectPath='/home/zjs/projects/ImageRetrieval/deepbit/examples/deepbit-lpss-32'
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
os.chdir(caffe_root)
def binaryproto2npy(meanFilePath):
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(meanFilePath, 'rb').read())
    mean_npy = caffe.io.blobproto_to_array(mean_blob)
    return mean_npy[0]

def MySort(List):
    Index = list(range(len(List)))
    Index.sort(key=lambda i:(List[i]))
    sortedList = sorted(List, reverse=False)
    return [sortedList, Index]

def count_distance(train_binary, test_binary, type='Hamming'):
    list=[]
    if type == 'Hamming':
        for train_binary_tmp in train_binary:
            smstr = np.nonzero(train_binary_tmp - test_binary)
            # print(smstr)
            sm = np.shape(smstr[0])[0]/test_binary.shape[1]
            list.append(sm)
    elif type == 'Euclidean':
        for train_binary_tmp in train_binary:
            op = np.linalg.norm(train_binary_tmp - test_binary)/test_binary.shape[1]
            list.append(op)

    [distance, index] = MySort(list)
    return [distance, index]

def featureExtract(image_path, net):
    # 读取图片并 resize 到256
    im = cv2.imread(image_path)
    im_32F = im.astype('float32')
    image = cv2.resize(im_32F, (256, 256), interpolation=cv2.INTER_AREA)
    # 转换为 3 256 256 结构， 减均值
    tmp = image.astype(np.float32, copy=False).transpose(2, 0, 1)
    blob = cv2.subtract(tmp, mean_npy.astype(np.float32, copy=False))
    # crop 224x224 image
    crop = blob[:, 16:240, 16:240]

    # net.blobs['data'].data[...] = transformer.preprocess('data', train_crop)
    net.blobs['data'].data[...] = crop

    # forward deepbit net & save the output
    out = net.forward()
    # train_binary[:, num_train] = (net.blobs['fc8_kevin'].data[0].flatten()).tolist()
    fc8_kevin = out['fc8_kevin'].reshape(1, 32)
    return fc8_kevin

### 设置参数
net_file=projectPath+'/deploy32.prototxt'
caffe_model=projectPath+'/DeepBit32_final_iter_1.caffemodel'
mean_file=caffe_root+'/data/ilsvrc12/imagenet_mean.binaryproto'
caffe.set_mode_gpu()
net = caffe.Net(net_file, caffe_model, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2, 0, 1))
mean_npy = binaryproto2npy(mean_file)

# data path
basePath = caffe_root + '/data/lpss/'

train_list_file = basePath + 'train-file-list.txt'
test_list_file = basePath + 'validation/positive.txt'
train_binary_file = os.path.join(caffe_root, 'python/lpss/train_data.bin')
####

if not os.path.exists(train_binary_file):
    file_train = open(train_list_file, 'r')
    train_paths = file_train.read().split('\n')
    num_train = 0
    # train_binary = np.array([[]])
    for train_path in train_paths:
        if train_path != '':
            fc8_kevin = featureExtract(caffe_root + train_path, net)
            if num_train:
                train_binary = np.vstack((train_binary, fc8_kevin))
            else:
                train_binary = fc8_kevin.copy()
            num_train += 1
            print(num_train)
    # np.savetxt(train_binary_file, train_binary)
    print(" train images num = ", num_train)
    train_binary.tofile(train_binary_file)
    file_train.close()

train_binary = np.fromfile(train_binary_file, dtype=np.float32).reshape(-1, 32)
# 求全局均值，并以均值將 train_binary 二值化
mean_th = np.mean(train_binary)
train_binary = (train_binary > mean_th) + 0

file_train = open(train_list_file, 'r')
train_paths = file_train.read().split('\n')
file_test = open(test_list_file, 'r')
test_paths = file_test.read().split('\n')
test_num = 0
for test_path in test_paths:
    print('test image :', test_num)
    # 读取图片并 resize 到256
    fc8_kevin = featureExtract(caffe_root + test_path, net)
    # 将 测试输出 编码二值化
    test_binary = (fc8_kevin > mean_th) + 0
    # 计算测试样本与训练样本集间的距离
    [distance, index] = count_distance(train_binary, test_binary, 'Hamming')

    # 保存与测试图相似度最高的10张图
    query_path = caffe_root + test_path
    query = cv2.resize(cv2.imread(query_path), (224, 224), interpolation=cv2.INTER_AREA)
    if query.shape[2] == 1:
        query = cv2.cvtColor(query, cv2.COLOR_GRAY2BGR)
    mean_distance = 0
    for i in range(10):
        retrieval = cv2.resize(cv2.imread(caffe_root + train_paths[index[i]]), (224, 224), interpolation=cv2.INTER_AREA)
        if retrieval.shape[2] == 1:
            retrieval = cv2.cvtColor(retrieval, cv2.COLOR_GRAY2BGR)

        # 打印 距离到图片上
        text = str(distance[i])
        cv2.putText(retrieval, text, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        query = cv2.hconcat((query, retrieval))
        if i < 3:
            mean_distance += distance[i]
    mean_distance = mean_distance / 3
    if mean_distance > 0.125:
        if not os.path.exists(caffe_root+'/analysis/lpss-val/positive'):
            os.makedirs(caffe_root+'/analysis/lpss-val/positive')
        cv2.imwrite(caffe_root+'/analysis/lpss-val/positive/'+str(test_num)+'.jpg', query)
    else:
        if not os.path.exists(caffe_root+'/analysis/lpss-val/negative'):
            os.makedirs(caffe_root+'/analysis/lpss-val/negative')
        cv2.imwrite(caffe_root+'/analysis/lpss-val/negative/'+str(test_num)+'.jpg', query)
    # cv2.imwrite(caffe_root+'/analysis/lpss-val/'+str(test_num)+'.jpg', query)
    # cv2.namedWindow('query', 1)
    # cv2.imshow('query', query)
    # cv2.waitKey()
    test_num += 1
# np.savetxt(train_binary_file, train_binary)
file_train.close()
