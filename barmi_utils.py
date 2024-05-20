import numpy as np
import imgproc
import cv2
def find_height_width_anomalies(widths, heights, threshold=2):
    """
    너비와 높이 배열에서 이상치를 찾아 각각의 인덱스를 반환하는 함수.
    
    :param widths: 너비 배열
    :param heights: 높이 배열
    :param threshold: 평균에서 벗어난 표준편차의 수를 기준으로 이상치를 판단, 기본값=2(표준편차)
    :return: (너비 이상치 인덱스 리스트, 높이 이상치 인덱스 리스트)
    """
    width_anomalies = []
    height_anomalies = []

    # 너비와 높이의 평균 및 표준편차 계산
    width_mean, width_std = np.mean(widths), np.std(widths)
    height_mean, height_std = np.mean(heights), np.std(heights)

    # 너비 이상치 탐지
    for i, w in enumerate(widths):
        if abs(w - width_mean) > threshold * width_std:
            width_anomalies.append(i)

    # 높이 이상치 탐지
    for i, h in enumerate(heights):
        if abs(h - height_mean) > threshold * height_std:
            height_anomalies.append(i)

    return width_anomalies, height_anomalies

# TODO: 평행선 이상 감지
# def find_align_anomalies(cores):
    # 


def text_similarity(ans_img,test_img):
    # 분석할 이미지를 로드하고 14등분하여 특징을 추출합니다.
    ans_divided_images = imgproc.divideImage(ans_img)
    test_divided_images = imgproc.divideImage(test_img)

    # 각 등분된 이미지의 특징을 추출하고, 기준 이미지의 특징과 유사성을 비교합니다.
    similarities = []
    # ans_features = []
    # test_features = []

    for i in range(len(ans_divided_images)):
        ans_features = imgproc.extractFeatures(ans_divided_images[i])
        test_features = imgproc.extractFeatures(test_divided_images[i])
        similarity = imgproc.compareFeatures(ans_features, test_features)
        similarities.append(similarity)
        # print(f"Segment {i+1} similarity: {similarity}")

    # 결과를 출력합니다.
    print("Similarities for each segment:", similarities)

# def loadImage(image_path):
#     # 이미지를 로드합니다.
#     image = cv2.imread(image_path)
#     return image