"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img
 
def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def divideImage(image, divisions=16):
    # 이미지를 가로로 등분하기
    height, width = image.shape[:2]
    division_width = width // divisions
    divided_images = [image[:, i*division_width:(i+1)*division_width] for i in range(divisions)]
    return divided_images

def extractFeatures(image):
    # 간단한 예시: 이미지에서 간단한 특징 추출 (여기서는 더 복잡한 특징 추출 방법을 적용할 수 있음)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    features = np.histogram(edges, bins=30)[0]
    return features

def compareFeatures(feature1, feature2):
    # 특징 간 유사성 측정 (여기서는 간단한 유클리디안 거리를 사용)
    distance = np.linalg.norm(feature1 - feature2)
    return distance


# def extractFeatures(image):
#     # SIFT 객체 생성
#     sift = cv2.xfeatures2d.SIFT_create()
#     # sift = cv2.SIFT_create()
    
#     # 이미지를 그레이스케일로 변환
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # SIFT를 사용하여 키포인트와 디스크립터 추출
#     keypoints, descriptors = sift.detectAndCompute(gray, None)
    
#     return descriptors

# def compareFeatures(feature1, feature2):
#     # BFMatcher 객체 생성
#     bf = cv2.BFMatcher()
    
#     # 두 디스크립터 집합 간의 가장 좋은 매칭을 찾음
#     matches = bf.knnMatch(feature1, feature2, k=2)
    
#     # Lowe's ratio test를 사용하여 좋은 매칭만 필터링
#     good_matches = []
#     for m,n in matches:
#         if m.distance < 0.75*n.distance:
#             good_matches.append([m])
    
#     # 좋은 매칭의 개수를 유사성 점수로 사용
#     similarity_score = len(good_matches)
#     return similarity_score