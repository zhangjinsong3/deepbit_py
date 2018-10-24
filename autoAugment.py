# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:05:38 2017

@author: 01376022

利用 图像检索 deepbit 方法自动增补数据库中稀少的数据

"""
import numpy as np
import re
import sys, os
import os.path
import argparse
import shutil
#设置当前目录
caffe_root = '/home/zjs/projects/ImageRetrieval/deepbit'
projectPath = '/home/zjs/projects/ImageRetrieval/deepbit/examples/deepbit-lpss-32'
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
os.chdir(caffe_root)
### 设置参数
net_file = projectPath+'/deploy32.prototxt'
caffe_model = projectPath+'/DeepBit32_final_iter_1.caffemodel'
mean_file = caffe_root+'/data/ilsvrc12/imagenet_mean.binaryproto'
caffe.set_mode_gpu()
net = caffe.Net(net_file, caffe_model, caffe.TEST)
# data path
basePath = '/media/zjs/A638551F3854F033/ImageRetrieval'

train_list_file = os.path.join(basePath, 'deepbit/train.txt')
train_binary_file = os.path.join(basePath, 'deepbit/train_data.bin')
####
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

def featureExtract(image_path, net, mean_np):
    if type(image_path) == str:
        # 读取图片并 resize 到256
        im = cv2.imread(image_path)
    else:
        # 输入为已经读取好的图片，numpy格式
        im = image_path
    im_32F = im.astype('float32')
    image = cv2.resize(im_32F, (256, 256), interpolation=cv2.INTER_AREA)
    # 转换为 3 256 256 结构， 减均值
    tmp = image.astype(np.float32, copy=False).transpose(2, 0, 1)
    blob = cv2.subtract(tmp, mean_np)
    # crop 224x224 image
    crop = blob[:, 16:240, 16:240]

    # net.blobs['data'].data[...] = transformer.preprocess('data', train_crop)
    net.blobs['data'].data[...] = crop

    # forward deepbit net & save the output
    out = net.forward()
    # train_binary[:, num_train] = (net.blobs['fc8_kevin'].data[0].flatten()).tolist()
    fc8_kevin = out['fc8_kevin'].reshape(1, 32)
    return fc8_kevin


def main(args):
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    mean_np = binaryproto2npy(mean_file).astype(np.float32, copy=False)

    if not os.path.exists(train_binary_file):
        file_train = open(train_list_file, 'r')
        train_paths = file_train.read().split('\n')
        num_train = 0
        # train_binary = np.array([[]])
        for train_path in train_paths:
            if train_path != '':
                fc8_kevin = featureExtract(os.path.join(basePath, train_path), net, mean_np)
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

    file_train = open(train_list_file, 'r+')
    train_paths = file_train.read().split('\n')[:-1]
    # 判断是文件夹中包含多个视频，还是单个视频
    videos_path = []
    video_dir = args.video_dir
    if os.path.isdir(video_dir):
        videos_path = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                     if (os.path.isfile(os.path.join(video_dir, f))
                         and (os.path.splitext(f)[1] == '.mp4' or os.path.splitext(f)[1] == '.avi'))]
        videos_path.sort()
    elif args.video_dir.lower().endswith(".mp4") or args.video_dir.lower().endswith(".avi"):
        videos_path = [args.video_dir]
    else:
        print('Invalid input format!')
    for video_path in videos_path:
        cap = cv2.VideoCapture(video_path)
        rval = cap.isOpened()
        if not rval:
            print(video_path, 'is not a invalid video!')
            return 0
        # 帧号
        frame_num = 0
        # 为每个视频创建文件夹，保存该视频挑选出的增补数据
        video_name = os.path.splitext(video_path)[0]
        images_save = os.path.join(args.save_dir, video_name)
        if not os.path.exists(images_save):
            os.makedirs(images_save)
        # 记录增补的图片数量
        add_num = 0
        while rval:
            rval, frame = cap.read()
            frame_num += 1
            if frame_num % 6 == 0:
                # 读取图片并 resize 到256
                fc8_kevin = featureExtract(frame, net, mean_np)
                # 将 测试输出 编码二值化
                test_binary = (fc8_kevin > mean_th) + 0
                # 计算测试样本与训练样本集间的距离
                [distance, index] = count_distance(train_binary, test_binary, 'Hamming')

                # 将与数据集中相似度不高的图片更新到数据库
                query = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                if query.shape[2] == 1:
                    query = cv2.cvtColor(query, cv2.COLOR_GRAY2BGR)
                mean_distance = 0
                for i in range(3):
                    retrieval = cv2.resize(cv2.imread(os.path.join(basePath, train_paths[index[i]])), (224, 224),
                                           interpolation=cv2.INTER_AREA)
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
                    # 增补此张图片到数据库中
                    new_path = os.path.join(images_save, '%06d.jpg' % frame_num)
                    cv2.imwrite(new_path, frame)
                    train_binary = np.vstack((train_binary, test_binary))
                    file_train.write(new_path.replace(basePath + '/', '') + '\n')
                    train_paths.append(new_path.replace(basePath + '/', ''))
                    add_num = add_num + 1
                # else:
                #     if not os.path.exists(caffe_root + '/analysis/lpss-val/negative'):
                #         os.makedirs(caffe_root + '/analysis/lpss-val/negative')
                #     cv2.imwrite(caffe_root + '/analysis/lpss-val/negative/' + str(test_num) + '.jpg', query)
                # cv2.imwrite(caffe_root+'/analysis/lpss-val/'+str(test_num)+'.jpg', query)

                # print add_num each hour
                if frame_num % (6*60*60) == 0:
                    print("add %4d images to the data base in the %2dth hour!" % (add_num, frame_num/(6*60*60)))
                    add_num = 0
                cv2.namedWindow('query', 1)
                cv2.imshow('query', query)
                cv2.waitKey(30)
    file_train.close()


    print("done")


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--image_width', default=640, type=int)
    parser.add_argument('--image_height', default=448, type=int)
    parser.add_argument('--video_dir',
                        default='/media/zjs/A638551F3854F033/ImageRetrieval/dailyvideo', type=str)
    parser.add_argument('--save_dir',
                        default='/media/zjs/A638551F3854F033/ImageRetrieval/deepbit/augmentData', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())