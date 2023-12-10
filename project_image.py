import cv2
import cvlib as cv
import numpy as np
import dlib

# dlib 얼굴 감지기와 랜드마크 감지기 불러오기
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# overlay 이미지 불러오기
image_right_eye_male = cv2.imread('./image/right_eye.png', cv2.IMREAD_UNCHANGED)
image_left_eye_male = cv2.imread('./image/left_eye.png', cv2.IMREAD_UNCHANGED)
image_nose_male = cv2.imread('./image/nose.png', cv2.IMREAD_UNCHANGED)

image_right_eye_female = cv2.imread('./image/right_eye_w.png', cv2.IMREAD_UNCHANGED)
image_left_eye_female = cv2.imread('./image/left_eye_w.png', cv2.IMREAD_UNCHANGED)
image_nose_female = cv2.imread('./image/nose_w.png', cv2.IMREAD_UNCHANGED)

def overlay(image, x, y, w, h, overlay_image):
    overlay_image = cv2.resize(overlay_image, (w//2, h//2))
    overlay_h, overlay_w = overlay_image.shape[:2]
    mask_image = overlay_image[:, :, 3] / 255
    for c in range(0, 3):
        image[y:y + overlay_h, x:x + overlay_w, c] = (overlay_image[:, :, c] * mask_image) + (image[y:y + overlay_h, x:x + overlay_w, c] * (1 - mask_image))

