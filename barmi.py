# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import tempfile

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from werkzeug.utils import secure_filename
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
import ocr_service
from collections import OrderedDict
CHARS_OF_LINE = 16

MODEL_PATH = "model/craft_mlt_25k.pth"
TEST_FOLDER = "sample"
CUDA_OPTION = False # For inference

STANDARD_SCORE = 25
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
        center_y = int((box[-1][1] + box[0][1]) / 2)
        centers.append((center_x, center_y))
        widths.append(width)
        heights.append(height)
    
    return num_boxes, centers, widths, heights
    

def get_coordination(index, centers, widths, heights) -> list:
    return (centers[index][0] + (widths//2),centers[index][1] - (heights//2))


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

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        print(image_path)
        image = imgproc.loadImage(image_path)

        bboxes, polys= test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, CUDA_OPTION, args.poly, refine_net)
        # print(len(polys),polys)
        polys = barmi_utils.merge_boxes(polys)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
        
        # Text dectection
        # print(filename,text_info(polys))
        nnum_boxes, ccenters, wwidths, hheights = text_info(polys)
    
    

    # ans_image = cv2.imread("korsong/wi15.png")
    # test_image = cv2.imread("korsong/songofkorea.png")
    # test_image = cv2.imread("korsong/trash.png")
    # barmi_utils.text_similarity(ans_image,test_image)

    print("elapsed time : {}s".format(time.time() - t))


def feedback(text_line, font_img, user_writing, handwriting_photo_path) -> list:
    texts_with_blank = list(text_line)  # "동해물과 백두산이" to ['동', '해', '물', '과', ' ', '백', '두', '산', '이']
    # string = "대한 사람 대한으로 길이 보전"
    # 공백을 제거한 문자 리스트와 원래 자리의 인덱스를 저장할 리스트 초기화
    char_list = []  # ['대', '한', '사', '람', '대', '한', '으', '로', '길', '이', '보', '전']
    original_indices = []  # [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]

    # 각 문자를 확인하면서 공백이 아닌 경우에만 리스트에 추가
    for index, char in enumerate(texts_with_blank):
        if char != ' ':
            char_list.append(char)
            original_indices.append(index)

    ans_image = imgproc.loadImage(font_img)
    user_image = imgproc.loadImage(user_writing)
    if ans_image is None:
        print("Error loading: ", font_img)
        return None
    if user_image is None:
        print("Error loading: ", user_writing)
        return None
    
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(MODEL_PATH, map_location='cpu')))
    net.eval()

    # refine_net = None
    # 2. text detection for user_writing
    bboxes, polys = test_net(net, user_image, 0.7, 1.0, 0.4, CUDA_OPTION, False, None)
    polys = barmi_utils.merge_boxes(polys)
    feedbacks = []
    # 3.POST PROCESSING
    user_boxes_num, user_centers, user_widths, user_heights = text_info(polys)
    print("user_heights", user_heights)
    print("centers", user_centers)
    if len(user_centers) != len(char_list):
        print("글자 수가 일치하지 않습니다.인식된 글자 수:", user_boxes_num, "정답 글자 수:", len(char_list))
        response = {
            "message": "error",
            "error": "글자 수가 일치하지 않습니다.",
            "feedbacks": feedbacks
        }
        return response
    # 4.Make feedback
    # 4-1. 글씨 크기가 작거나 큰 경우
    large_anomalies, small_anomalies = barmi_utils.find_area_anomalies(user_widths, user_heights)
    for index in large_anomalies:
        x, y = get_coordination(index, user_centers, user_widths[index], user_heights[index])
        feedbacks = add_or_merge_feedback(feedbacks, "글씨 크기가 너무 큽니다.", x, y)
    for index in small_anomalies:
        x, y = get_coordination(index, user_centers, user_widths[index], user_heights[index])
        feedbacks = add_or_merge_feedback(feedbacks, "글씨 크기가 너무 작습니다.", x, y)
    # 4-2. 평행선 이상 감지
    align_anomalies = barmi_utils.find_align_anomalies(user_centers)
    for index in align_anomalies:
        x, y = get_coordination(index, user_centers, user_widths[index], user_heights[index])
        feedbacks = add_or_merge_feedback(feedbacks, "글씨가 평행선에 맞춰져 있지 않습니다.", x, y)
    # TODO : 띄어쓰기 감지
    # 4-3. 인식 결과가 다른 경우
    recognization_texts = ocr_service.ocr_api(handwriting_photo_path)
    print("recognization_texts", recognization_texts)
    recognization_texts = list(recognization_texts)
    actual_string = ''.join(char_list)
    # char_diff[index] -> ["ㄱ","ㅏ","ㄴ"] OR []
    char_diff = barmi_utils.find_line_diffrence(actual_string, recognization_texts)
    for i in range(len(char_diff)):
        if char_diff[i]:
            x, y = get_coordination(i, user_centers, user_widths[i], user_heights[i])
            need_to_modify = ",".join(char_diff[i])
            feedbacks = add_or_merge_feedback(feedbacks, f"글자{need_to_modify}을 다시 작성해보세요.", x, y)
    
    response = {
        "message": "success",
        "feedbacks": feedbacks
    }
    return response

def add_or_merge_feedback(feedbacks, feedback_text, x, y) -> list:
    """
    Add feedback to the list if coordinates are unique,
    or merge feedbacks if the coordinates already exist.
    """
    for feedback in feedbacks:
        if feedback['coordinates']['x'] == x and feedback['coordinates']['y'] == y:
            feedback['feedback'] += " " + feedback_text
            return feedbacks
    feedbacks.append({
        "feedback": feedback_text,
        "coordinates": {"x": x, "y": y}
    })
    return feedbacks



def game(text_line, font_img, user_writing, handwriting_photo_path) -> int:


    texts_with_blank = list(text_line)  # "동해물과 백두산이" to ['동', '해', '물', '과', ' ', '백', '두', '산', '이']
    # string = "대한 사람 대한으로 길이 보전"
    # 공백을 제거한 문자 리스트와 원래 자리의 인덱스를 저장할 리스트 초기화
    char_list = []  # ['대', '한', '사', '람', '대', '한', '으', '로', '길', '이', '보', '전']
    original_indices = []  # [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]

    # 각 문자를 확인하면서 공백이 아닌 경우에만 리스트에 추가
    for index, char in enumerate(texts_with_blank):
        if char != ' ':
            char_list.append(char)
            original_indices.append(index)
    # Score
    score = STANDARD_SCORE
    deduction_scores = [0] *len(char_list)

    # Load images
    ans_image = imgproc.loadImage(font_img)
    user_image = imgproc.loadImage(user_writing)
    if ans_image is None:
        print("Error loading: ", font_img)
        return None
    if user_image is None:
        print("Error loading: ", user_writing)
        return None
    
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(MODEL_PATH, map_location='cpu')))
    net.eval()

    # refine_net = None
    # 2. text detection for user_writing
    bboxes, polys = test_net(net, user_image, 0.7, 1.0, 0.4, CUDA_OPTION, False, None)
    polys = barmi_utils.merge_boxes(polys)
    # 3.POST PROCESSING
    user_boxes_num, user_centers, user_widths, user_heights = text_info(polys)
    print("user_heights", user_heights)
    print("centers", user_centers)
    if len(user_centers) != len(char_list):
        print("글자 수가 일치하지 않습니다.인식된 글자 수:", user_boxes_num, "정답 글자 수:", len(char_list))
        response = {
            "message": "error",
            "error": "글자 수가 일치하지 않습니다.",
            "score": 0
        }
        return response
    # 4.Make feedback
    # 4-1. 글씨 크기가 작거나 큰 경우
    large_anomalies, small_anomalies = barmi_utils.find_area_anomalies(user_widths, user_heights)
    for index in large_anomalies:
        deduction_scores[index] -= 5
    for index in small_anomalies:
        deduction_scores[index] -= 5
    # 4-2. 평행선 이상 감지
    align_anomalies = barmi_utils.find_align_anomalies(user_centers)
    for index in align_anomalies:
        deduction_scores[index] -= 5
    # TODO : 띄어쓰기 감지
    # 4-3. 인식 결과가 다른 경우
    recognization_texts = ocr_service.ocr_api(handwriting_photo_path)
    print("recognization_texts", recognization_texts)
    recognization_texts = list(recognization_texts)
    actual_string = ''.join(char_list)
    # char_diff[index] -> ["ㄱ","ㅏ","ㄴ"] OR []
    char_diff = barmi_utils.find_line_diffrence(actual_string, recognization_texts)
    for i in range(len(char_diff)):
        if char_diff[i]:
            deduction_scores[i] -= len(char_diff[i])
    # Calculate score
    score = max(0, score + sum(deduction_scores))    
    response = {
        "message": "success",
        "score": score
    }
    return response


