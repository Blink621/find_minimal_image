# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:33:51 2019

@author: Blink
"""

import torch
from PIL import Image

import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float

import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from torchvision import transforms
from dnnbrain.dnn.io import NetLoader
from dnnbrain.dnn.analyzer import dnn_activation

def imageTotensor(image):
    """
    输入一张图片 
    输出以这张图片为基础的4Dtensor
    """
    img = transforms.ToTensor()(image)  # 将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且/255归一化到[0,1.0]之间
    img_tensor = torch.reshape(img,[1,3,227,227])
    return img_tensor 

def noisyimage(image):
    image_copy=image.copy() #复制原图以供转换操作,否则图片会发生改变
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            color = (255,255,255)
            image_copy[i,j] = color
    return image_copy
    
def dnn_segmentation(image,net,layer,channel):
    """
    输入指定路径的图片，以watershed方法超像素处理后每个superpixel被拿出来的图片,
    按照激活从大到小排序，再拼回去直到激活达到最大值
    背景为高斯噪声
    ————————————
    变量
    image：类型为numpy array
    net：str
    layer：str
    channel：int 
    """
    act=[]
    net_loader = NetLoader(net)
    
    img = img_as_float(image)
    noise = noisyimage(image)
    
    
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")  #分割图片
    plt.imshow((mark_boundaries(img, segments_fz)))
    plt.axis('off')
        
    
    for label in range(np.min(segments_fz),np.max(segments_fz)+1):
        noise_copy = noise.copy()#生成噪声背景
        location = np.where(segments_fz==label)  #通过标记找出每个patch的位置
        for i in range(len(location[0])):
            noise_copy[location[0][i],location[1][i]]=image[location[0][i],location[1][i]] #改变segmentation对应原始图片位置的RGB值
            
        noise_tensor = imageTotensor(noise_copy) #生成噪音图片并转换为tensor    
        dnn_acts= dnn_activation(noise_tensor, net_loader.model, net_loader.layer2keys[layer])  # 提取激活
        Ncolumn = np.sum(dnn_acts[0,channel])   #act为所有分割后的图片的激活 顺序为label的标记值
        act.append(Ncolumn)
    
    act_array=np.array(act)
    act_sorted=np.argsort(-act_array)  #按激活从大到小的顺序得出索引值
    
    act_max = act[act_sorted[0]]  #找到初始的最大激活的patch
    act_max_sorted=[]
    
    noise_copy_a = noise.copy()#生成噪声背景
    for index in act_sorted:
        locaton_sorted = np.where(segments_fz == index)
        for j in range(len(locaton_sorted[0])):
            noise_copy_a[locaton_sorted[0][j],locaton_sorted[1][j]] = image[locaton_sorted[0][j],locaton_sorted[1][j]] #改变segmentation对应原始图片位置的RGB值
        noise_sorted_tensor = imageTotensor(noise_copy_a)
        act_noise = dnn_activation(noise_sorted_tensor, net_loader.model, net_loader.layer2keys[layer])
        act_max = np.sum(act_noise[0,channel])
        act_max_sorted.append(act_max)
    
    
    act_pic = max(act_max_sorted)
    
    act_max_index = act_max_sorted.index(act_pic)
    act_back_index = act_sorted[act_max_index] #!!!!
    
    
    for index_a in act_sorted:
        locaton_sorted_a = np.where(segments_fz == index_a)
        for p in range(len(locaton_sorted_a[0])):
            noise[locaton_sorted_a[0][p],locaton_sorted_a[1][p]] = image[locaton_sorted_a[0][p],locaton_sorted_a[1][p]] #改变segmentation对应原始图片位置的RGB值
        if index_a == act_back_index:
            break
    
    return noise
            

image_original = Image.open('#add your path')
image_RGB  = image_original.convert("RGB") 
image = np.asarray(image_RGB)


net   = 'alexnet'
layer = '#add layer you want'
channel = #add channel you want

net_loader = NetLoader(net) #载入网络
img_org_tensor = imageTotensor(image)
act_org = dnn_activation(img_org_tensor, net_loader.model, net_loader.layer2keys[layer])
act_org_sum = np.sum(act_org[0,channel]) #计算原始激活

image_min = dnn_segmentation(image, net, layer, channel)  # 提取激活

print(act_org_sum)
plt.imshow(image_min)

    
                
