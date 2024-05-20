# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import barmi_utils
from craft import CRAFT

from collections import OrderedDict
CHARS_OF_LINE = 16

# python test.py --cuda=False --trained_model=model/craft_mlt_25k.pth --test_folder=sample --link_threshold=1.0
MODEL_PATH = "model/craft_mlt_25k.pth"
TEST_FOLDER = "sample"
CUDA_OPTION = False

# TODO: LINK threshold

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    # print("poly!!!!!",polys[0],polys[1])
    t1 = time.time() - t1
    
    # render results (optional)
    render_img = score_text.copy()
    # render_img = np.hstack((render_img, score_link))
    # ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys


def text_info(boxes):
    # np.sort(boxes,axis=0)
    # boxes = np.sort(boxes)
    boxes = sorted(boxes, key=lambda x: x[0, 0])
    num_boxes = len(boxes)
    widths = []
    heights = []
    centers = []
    
    for box in boxes:
        width = int(box[1][0] - box[0][0])
        height = int(box[-1][1] - box[0][1])
        center_x = int((box[1][0] + box[0][0]) / 2)
        center_y = int((box[-1][1] - box[0][1]) / 2)
        centers.append((center_x, center_y))
        widths.append(width)
        heights.append(height)
    
    return num_boxes, centers, widths, heights
    


if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + MODEL_PATH + ')')
    if CUDA_OPTION:
        net.load_state_dict(copyStateDict(torch.load(MODEL_PATH)))
    else:
        net.load_state_dict(copyStateDict(torch.load(MODEL_PATH, map_location='cpu')))


    if CUDA_OPTION:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # # LinkRefiner
    # # TODO: 만약 refiner해야할 경우 추후에 처리

    refine_net = None
    # if args.refine:
    #     from refinenet import RefineNet
    #     refine_net = RefineNet()
    #     print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    #     if CUDA_OPTION:
    #         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
    #         refine_net = refine_net.cuda()
    #         refine_net = torch.nn.DataParallel(refine_net)
    #     else:
    #         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    #     refine_net.eval()
    #     args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys= test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, CUDA_OPTION, args.poly, refine_net)
        
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

        # Text dectection
        print(filename,text_info(polys))
        nnum_boxes, ccenters, wwidths, hheights = text_info(polys)
        width_anomalies, height_anomalies = barmi_utils.find_height_width_anomalies(wwidths, hheights)
        print("너비 이상치 인덱스:", width_anomalies)
        print("높이 이상치 인덱스:", height_anomalies)
        # print(bboxes)
        



    print("elapsed time : {}s".format(time.time() - t))
    

    ans_image = cv2.imread("korsong/songofkorea_ans(2).png")
    test_image = cv2.imread("korsong/songofkorea.png")
    # test_image = cv2.imread("korsong/trash.png")
    if ans_image is None:
        print("Error loading: korsong/songofkorea_ans.png")
    if test_image is None:
        print("Error loading: korsong/songofkorea.png")
    # 이미지 사이즈 확인
    ans_height, ans_width = ans_image.shape[:2]
    test_height, test_width = test_image.shape[:2]

    # 이미지 사이즈 비교 및 조정
    if ans_height != test_height or ans_width != test_width:
        # 두 이미지 중 더 작은 사이즈를 찾음
        new_width = min(ans_width, test_width)
        new_height = min(ans_height, test_height)
        
        # 더 작은 사이즈에 맞춰서 두 이미지 모두 조정
        ans_image = cv2.resize(ans_image, (new_width, new_height))
        test_image = cv2.resize(test_image, (new_width, new_height))
    barmi_utils.text_similarity(ans_image,test_image)

