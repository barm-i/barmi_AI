import numpy as np
import imgproc
import cv2
from skimage.metrics import structural_similarity as ssim

CHO = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 
    'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]
JUNG = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 
    'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]
JONG = [
    '', 'ㄱ','ㄲ','ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 
    'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 
    'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

def find_letter_diffrence(ans_letter, test_letter):
    """
    Input : ans_letter, test_letter 한글 글자
    Output : diffrence 초성, 중성, 종성(optional)의 차이 값, 없으면 빈 리스트 반환
    """
    ans_segments = korean_segmentation(ans_letter)
    test_segments = korean_segmentation(test_letter)
    diff = []
    if ans_letter == test_letter:
        return diff
    if len(test_segments) == 2:
        test_segments.append("")
    if len(ans_segments) != 3:
        return ans_segments
    for i in range(len(ans_segments)):
        if ans_segments[i] != test_segments[i]:
            diff.append(ans_segments[i])
    return diff

def find_line_diffrence(ans_line, user_line):
    """
    Input : ans_line, user_line 한글 글자
    Output : diffrence 초성, 중성, 종성(optional)의 차이 값, 없으면 빈 리스트 반환
    """
    diff = []
    for i in range(len(ans_line)):
        ans_letter = ans_line[i]
        user_letter = user_line[i]
        diff.append(find_letter_diffrence(ans_letter, user_letter))
    return diff


def korean_segmentation(letter):
    """
    Input : letter 한글 글자
    Output : segmentation 초성 중성 종성(optional)
    """
    segementation = []
    if ord("가") <= ord(letter) <= ord("힣"):
        index = ord(letter) - ord("가")
        cho = int((index / 28) / 21)
        jung = int((index / 28) % 21)
        jong = int(index % 28)
        segementation.append(CHO[cho])
        segementation.append(JUNG[jung])
        if jong > 0:
            segementation.append(JONG[jong])
    else:
        segementation.append(letter)
    return segementation



def letter_type(segments):
    """
    Input : segments 초성 중성 종성(optional)
    Output : type of letter -> 1: "이", 2: "으" 3: "와" 4: "잉" 5: "응" 6: "읭" 글자 크기에 따른 분류
    """
    letter_type = None
    if "" <= segments[1] <= "ㅕ" or segments[1] == "ㅣ":
        letter_type = 1
    elif segments[1] == "ㅡ" or segments[1] == "ㅜ" or segments[1] == "ㅗ" or segments[1] == "ㅛ" or segments[1] == "ㅠ":
        letter_type = 2
    else:
        letter_type = 3
    if len(segments) == 3:
        letter_type += 3
    return letter_type



def find_height_width_anomalies(widths, heights, threshold=2):
    """
    너비와 높이 배열에서 이상치를 찾아 각각의 인덱스를 반환하는 함수.
    
    :param widths: 너비 배열
    :param heights: 높이 배열
    :param threshold: 평균에서 벗어난 표준편차의 수를 기준으로 이상치를 판단, 기본값=2(표준편차)
    :return: (너비 이상치 인덱스 리스트, 높이 이상치 인덱스 리스트)
    """
    
    width_anomalies_large = []
    height_anomalies_large = []
    width_anomalies_small = []
    height_anomalies_small = []
    
    # 너비와 높이의 평균 및 표준편차 계산
    width_mean, width_std = np.mean(widths), np.std(widths)
    height_mean, height_std = np.mean(heights), np.std(heights)

    # 너비 이상치 탐지
    for i, w in enumerate(widths):
        if abs(w - width_mean) > threshold * width_std:
            if w > width_mean:
                width_anomalies_large.append(i)
            else:
                width_anomalies_small.append(i)
    # 높이 이상치 탐지
    for i, h in enumerate(heights):
        if abs(h - height_mean) > threshold * height_std:
            if h > height_mean:
                height_anomalies_large.append(i)
            else:
                height_anomalies_small.append(i)
    return width_anomalies_large, height_anomalies_large, width_anomalies_small, height_anomalies_small


def find_area_anomalies(widths, heights, threshold=2):
    """
    너비와 높이 배열에서 영역(너비*높이)의 이상치를 찾아 인덱스를 반환하는 함수.
    
    :param widths: 너비 배열
    :param heights: 높이 배열
    :param threshold: 평균에서 벗어난 표준편차의 수를 기준으로 이상치를 판단, 기본값=2(표준편차)
    :return: (영역 이상치 인덱스 리스트)
    """
    
    area_anomalies_large = []
    area_anomalies_small = []
    
    # 영역 계산
    areas = np.array(widths) * np.array(heights)
    
    # 영역의 평균 및 표준편차 계산
    area_mean, area_std = np.mean(areas), np.std(areas)
    
    # 영역 이상치 탐지
    for i, area in enumerate(areas):
        if abs(area - area_mean) > threshold * area_std:
            if area > area_mean:
                area_anomalies_large.append(i)
            else:
                area_anomalies_small.append(i)
                
    return area_anomalies_large, area_anomalies_small

def find_align_anomalies(centers, threshold=5, middle =25):
    """
    평행선 이상을 찾아 인덱스를 반환하는 함수.
    
    :param centers: 중심값 좌표 배열
    :param threshold: 기준 y 값
    :return: 이상치 인덱스 리스트
    """
    align_anomalies = []
    for i, center in enumerate(centers):
        y_value = center[1]
        if y_value > middle + threshold or y_value < middle-threshold:
            align_anomalies.append(i)
    return align_anomalies


def text_similarity(ans_img,test_img):
    # 분석할 이미지를 로드하고 16등분하여 특징을 추출합니다.
    ans_divided_images = imgproc.divideImage(ans_img)
    test_divided_images = imgproc.divideImage(test_img)

    # 각 등분된 이미지의 특징을 추출하고, 기준 이미지의 특징과 유사성을 비교합니다.
    similarities = []
    # ans_features = []
    # test_features = []

    for i in range(len(ans_divided_images)):
        # TODO: histogram, ssim 성능비교
        # histogram
        # ans_features = imgproc.extractFeatures(ans_divided_images[i])
        # test_features = imgproc.extractFeatures(test_divided_images[i])
        # similarity = imgproc.compareFeatures(ans_features, test_features)
        # ans_features = imgproc.extractFeatures(ans_divided_images[i])
        # test_features = imgproc.extractFeatures(test_divided_images[i])
        similarity = structural_similarity(ans_divided_images[i],test_divided_images[i])
        similarities.append(similarity)
        # print(f"Segment {i+1} similarity: {similarity}")

    # 결과를 출력합니다.
    print("Similarities for each segment:", similarities)

# def loadImage(image_path):
#     # 이미지를 로드합니다.
#     image = cv2.imread(image_path)
#     return image
def structural_similarity(img1, img2):
    """
    두 이미지의 구조적 유사도를 계산합니다.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 구조적 유사도 계산
    similarity = ssim(gray1, gray2)
    
    return similarity



def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    # Convert the boxes to the (x1, y1, x2, y2) format
    boxes = np.array([[
        box[0][0], box[0][1],  # x1, y1
        box[2][0], box[2][1]   # x2, y2
    ] for box in boxes])

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def merge_boxes(boxes):
    merged_boxes = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
        current_box = boxes[i]
        current_center_x = (current_box[0][0] + current_box[1][0]) / 2
        current_center_y = (current_box[0][1] + current_box[2][1]) / 2
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            next_box = boxes[j]
            next_center_x = (next_box[0][0] + next_box[1][0]) / 2
            next_center_y = (next_box[0][1] + next_box[2][1]) / 2
            if abs(current_center_x - next_center_x) <= 15:
                # Merge boxes
                min_x = min(current_box[0][0], next_box[0][0])
                max_x = max(current_box[1][0], next_box[1][0])
                min_y = min(current_box[0][1], next_box[0][1])
                max_y = max(current_box[2][1], next_box[2][1])
                current_box = np.array([
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y]
                ], dtype=np.float32)
                used[j] = True
        merged_boxes.append(current_box)
        used[i] = True
    
    return np.array(merged_boxes, dtype=object)