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


# 사진 불러오기
img = cv2.imread('./image/woman_face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 감지
faces = detector(gray)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()

    # 얼굴 roi 지정
    face_crop = np.copy(img[y1:y2, x1:x2])

    # 성별 예측하기
    label, confidence = cv.detect_gender(face_crop)

    idx = np.argmax(confidence)
    label = label[idx]

    # 랜드마크 감지
    landmarks = predictor(gray, face)

    # 눈과 코의 위치를 감지하여 overlay 이미지 적용
    for n in range(36, 48):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        if n < 42:  # 왼쪽 눈 위치
            if label == 'male':
                overlay(img, x, y, x2-x1, y2-y1, image_left_eye_male)
            else:
                overlay(img, x, y, x2-x1, y2-y1, image_left_eye_female)
        elif n < 48:  # 오른쪽 눈 위치
            if label == 'male':
                overlay(img, x, y, x2-x1, y2-y1, image_right_eye_male)
            else:
                overlay(img, x, y, x2-x1, y2-y1, image_right_eye_female)
        else:  # 코 위치
            if label == 'male':
                overlay(img, x, y, x2-x1, y2-y1, image_nose_male)
            else:
                overlay(img, x, y, x2-x1, y2-y1, image_nose_female)