import cv2
import cvlib as cv
import numpy as np
import dlib
import random

# dlib 얼굴 감지기와 랜드마크 감지기 불러오기
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# # overlay 이미지 불러오기
image_right_eye_male = cv2.imread('./image/right_eye_1.png', cv2.IMREAD_UNCHANGED)
image_left_eye_male = cv2.imread('./image/left_eye_1.png', cv2.IMREAD_UNCHANGED)
image_nose_male = cv2.imread('./image/nose_1.png', cv2.IMREAD_UNCHANGED)

image_right_eye_female = cv2.imread('./image/right_eye_w.png', cv2.IMREAD_UNCHANGED)
image_left_eye_female = cv2.imread('./image/left_eye_w.png', cv2.IMREAD_UNCHANGED)
image_nose_female = cv2.imread('./image/nose_w.png', cv2.IMREAD_UNCHANGED)

# overlay 이미지 불러오기 (이지우님)
image_right_eye_male_1 = cv2.imread('./image/horn_right.png', cv2.IMREAD_UNCHANGED)
image_left_eye_male_1 = cv2.imread('./image/horn_left.png', cv2.IMREAD_UNCHANGED)
image_nose_male_1 = cv2.imread('./image/Moodang_1.png', cv2.IMREAD_UNCHANGED)

image_right_eye_female_1 = cv2.imread('./image/snata_hat.png', cv2.IMREAD_UNCHANGED)
image_left_eye_female_1 = cv2.imread('./image/snata_hat.png', cv2.IMREAD_UNCHANGED)
image_nose_female_1 = cv2.imread('./image/Moodang_1.png', cv2.IMREAD_UNCHANGED)

# # 원래코드
# def overlay(image, x, y, w, h, overlay_image):
#     overlay_image = cv2.resize(overlay_image, (w//2, h//2))
#     overlay_h, overlay_w = overlay_image.shape[:2]
#     mask_image = overlay_image[:, :, 3] / 255
#     for c in range(0, 3):
#         image[y:y + overlay_h, x:x + overlay_w, c] = (overlay_image[:, :, c] * mask_image) + (image[y:y + overlay_h, x:x + overlay_w, c] * (1 - mask_image))

def place_overlay(image, x, y, overlay_image):
    # Calculate the size of the overlay image
    overlay_h, overlay_w, _ = overlay_image.shape
    
    # Calculate the start and end points, ensuring they are within the image bounds
    x_start, x_end = max(x, 0), min(x + overlay_w, image.shape[1])
    y_start, y_end = max(y, 0), min(y + overlay_h, image.shape[0])
    
    # Cut the overlay image if it goes out of the bounds
    overlay_image = overlay_image[:y_end-y_start, :x_end-x_start]
    
    # Make sure the overlay image is not empty
    if overlay_image.shape[0] > 0 and overlay_image.shape[1] > 0:
        alpha_s = overlay_image[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        # Overlay the image
        for c in range(0, 3):
            image[y_start:y_end, x_start:x_end, c] = (alpha_s * overlay_image[:, :, c] +
                                                      alpha_l * image[y_start:y_end, x_start:x_end, c])

# 사진 불러오기
img = cv2.imread('./image/woman_face.jpg')
# img = cv2.imread('./image/man_face_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 감지
faces = detector(gray)

# Function to calculate bounding boxes
def calculate_bounding_box(landmarks, start, end):
    xs = [landmarks.part(i).x for i in range(start, end)]
    ys = [landmarks.part(i).y for i in range(start, end)]
    return (min(xs), min(ys), max(xs), max(ys))

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()

    # 얼굴 roi 지정
    face_crop = np.copy(img[y1:y2, x1:x2])

    # 성별 예측하기
    label, confidence = cv.detect_gender(face_crop)

    idx = np.argmax(confidence)
    label = label[idx]

    # 성별 레이블을 이미지의 왼쪽 상단에 추가
    # cv2.putText(img, label, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 2)

    # 랜드마크 감지
    landmarks = predictor(gray, face)

    # # 눈과 코의 위치를 감지하여 overlay 이미지 적용
    # for n in range(36, 48):
    #     x = landmarks.part(n).x
    #     y = landmarks.part(n).y
    #     if n < 42:  # 왼쪽 눈 위치
    #         if label == 'male':
    #             overlay(img, x, y, x2-x1, y2-y1, image_left_eye_male)
    #         else:
    #             overlay(img, x, y, x2-x1, y2-y1, image_left_eye_female)
    #     elif n < 48:  # 오른쪽 눈 위치
    #         if label == 'male':
    #             overlay(img, x, y, x2-x1, y2-y1, image_right_eye_male)
    #         else:
    #             overlay(img, x, y, x2-x1, y2-y1, image_right_eye_female)
    #     else:  # 코 위치
    #         if label == 'male':
    #             overlay(img, x, y, x2-x1, y2-y1, image_nose_male)
    #         else:
    #             overlay(img, x, y, x2-x1, y2-y1, image_nose_female)

    # Calculate bounding boxes for the left and right eyes
    left_eye_box = calculate_bounding_box(landmarks, 36, 42)
    right_eye_box = calculate_bounding_box(landmarks, 42, 48)

    # # Draw bounding boxes for the eyes
    # cv2.rectangle(img, (left_eye_box[0], left_eye_box[1]), (left_eye_box[2], left_eye_box[3]), (0, 255, 0), 2)
    # cv2.rectangle(img, (right_eye_box[0], right_eye_box[1]), (right_eye_box[2], right_eye_box[3]), (0, 255, 0), 2)

    # Calculate bounding box for the nose using the correct landmark indices
    nose_box = calculate_bounding_box(landmarks, 27, 36)

    # # Draw bounding box for the nose
    # cv2.rectangle(img, (nose_box[0], nose_box[1]), (nose_box[2], nose_box[3]), (0, 255, 0), 2)

    # # Overlay images based on gender
    # if label == 'male':
    #     overlay(img, left_eye_box[0], left_eye_box[1], left_eye_box[2]-left_eye_box[0], left_eye_box[3]-left_eye_box[1], image_left_eye_male)
    #     overlay(img, right_eye_box[0], right_eye_box[1], right_eye_box[2]-right_eye_box[0], right_eye_box[3]-right_eye_box[1], image_right_eye_male)
    #     overlay(img, nose_box[0], nose_box[1], nose_box[2]-nose_box[0], nose_box[3]-nose_box[1], image_nose_male)
    # else:
    #     overlay(img, left_eye_box[0], left_eye_box[1], left_eye_box[2]-left_eye_box[0], left_eye_box[3]-left_eye_box[1], image_left_eye_female)
    #     overlay(img, right_eye_box[0], right_eye_box[1], right_eye_box[2]-right_eye_box[0], right_eye_box[3]-right_eye_box[1], image_right_eye_female)
    #     overlay(img, nose_box[0], nose_box[1], nose_box[2]-nose_box[0], nose_box[3]-nose_box[1], image_nose_female)
    
    number = random.choice([0, 1])
    # number = 1

    # Overlay images based on gender
    if label == 'male':
        if number == 0:
            place_overlay(img, left_eye_box[0], left_eye_box[1] - 200 - image_left_eye_female.shape[0], image_left_eye_male)
            place_overlay(img, right_eye_box[0], right_eye_box[1] - 200 - image_right_eye_female.shape[0], image_right_eye_male)
            nose_bottom_y = nose_box[3]
            place_overlay(img, nose_box[0] - 20, nose_bottom_y - image_nose_female.shape[0], image_nose_male)
        else:
            place_overlay(img, left_eye_box[0], left_eye_box[1] - 200 - image_left_eye_female.shape[0], image_left_eye_male_1)
            place_overlay(img, right_eye_box[0], right_eye_box[1] - 200 - image_right_eye_female.shape[0], image_right_eye_male_1)
            nose_bottom_y = nose_box[3]
            place_overlay(img, nose_box[0] - 20, nose_bottom_y - image_nose_female.shape[0], image_nose_male_1)
    else:
        if number == 0:
            place_overlay(img, left_eye_box[0], left_eye_box[1] - 200 - image_left_eye_female.shape[0], image_left_eye_female)
            place_overlay(img, right_eye_box[0], right_eye_box[1] - 200 - image_right_eye_female.shape[0], image_right_eye_female)
            nose_bottom_y = nose_box[3]
            place_overlay(img, nose_box[0] - 70, nose_bottom_y - image_nose_female.shape[0], image_nose_female)
        else:
            place_overlay(img, left_eye_box[0], left_eye_box[1] - 200 - image_left_eye_female.shape[0], image_left_eye_female_1)
            place_overlay(img, right_eye_box[0], right_eye_box[1] - 200 - image_right_eye_female.shape[0], image_right_eye_female_1)
            nose_bottom_y = nose_box[3]
            place_overlay(img, nose_box[0] - 20, nose_bottom_y - image_nose_female.shape[0], image_nose_female_1)


# 원하는 최대 너비를 설정합니다. 예를 들어, 800px로 설정했습니다.
desired_width = 800

# 원본 이미지의 너비와 높이를 가져옵니다.
original_width, original_height = img.shape[1], img.shape[0]

# 원본 이미지의 너비가 원하는 너비보다 큰 경우에만 크기를 조절합니다.
if original_width > desired_width:
    # 원본 이미지의 비율을 유지하면서 크기를 조절합니다.
    new_height = int(original_height * desired_width / original_width)
    img = cv2.resize(img, (desired_width, new_height))


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()   