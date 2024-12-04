import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 채널 축 설정 (grayscale 이미지의 경우 None, RGB 이미지의 경우 -1)
channel_axis = None


def ssim_score(true, pred):
    if true.shape[-1] == 1:
        true = true.reshape(true.shape[:-1])
        pred = pred.reshape(pred.shape[:-1])
        channel_axis = None
    else:
        channel_axis = -1
    # 전체 RGB 이미지를 사용해 SSIM 계산 (channel_axis=-1)
    ssim_value = ssim(
        true, pred, channel_axis=channel_axis, data_range=pred.max() - pred.min()
    )
    return ssim_value


def masked_ssim_score(true, pred, mask):
    if true.shape[-1] == 1:
        true = true.reshape(true.shape[:-1])
        pred = pred.reshape(pred.shape[:-1])
        mask = mask.reshape(mask.shape[:-1])
        multichannel = False
    else:
        multichannel = True
    # 손실 영역의 좌표에서만 RGB 채널별 픽셀 값 추출
    mask = (mask > 0)

    # 손실 영역 픽셀만으로 SSIM 계산 (채널축 사용)
    _, ssim_map = ssim(
        true,
        pred,
        multichannel=multichannel,
        data_range=pred.max() - pred.min(),
        full=True
    )

    masked_ssim = np.mean(ssim_map[mask])
    
    return masked_ssim


def histogram_similarity(true, pred):
    # BGR 이미지를 HSV로 변환
    true_hsv = cv2.cvtColor(true, cv2.COLOR_BGR2HSV)
    pred_hsv = cv2.cvtColor(pred, cv2.COLOR_BGR2HSV)

    # H 채널에서 히스토그램 계산 및 정규화
    hist_true = cv2.calcHist([true_hsv], [0], None, [180], [0, 180])
    hist_pred = cv2.calcHist([pred_hsv], [0], None, [180], [0, 180])
    hist_true = cv2.normalize(hist_true, hist_true).flatten()
    hist_pred = cv2.normalize(hist_pred, hist_pred).flatten()

    # 히스토그램 간 유사도 계산 (상관 계수 사용)
    similarity = cv2.compareHist(hist_true, hist_pred, cv2.HISTCMP_CORREL)
    return similarity
