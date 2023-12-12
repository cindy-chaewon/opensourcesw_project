# Facial Landmark Overlay with Gender Prediction
---
This Python script uses the dlib library to detect facial landmarks and predict the gender of a person in an input image.
It overlays gender-specific eye and nose images onto the detected facial landmarks. 
The script also includes functionality to resize the output image for better visualization.
There are 2 filters for men and 2 filters for women.
After detecting gender, two filters were randomly applied.

# Reference Code Sources and Changes
---
Our team referred to this lecture [python lecture - Image Processing (OpenCV)](https://www.inflearn.com/course/%EB%82%98%EB%8F%84%EC%BD%94%EB%94%A9-%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%B2%98%EB%A6%AC).
Additionally, we applied gender discrimination so that filters could be applied randomly.

## Prerequisites 
---

1. Python 3.11.5
2. OpenCV (cv2) (4.8.1.78)
3. cvlib(0.2.7)
4. dlib(19.24.2)

Install the required libraries using the following in anaconda:
- pip install opencv-python
- pip install cvlib
- pip install dlib

## Usage 
---

Place the input image in the ./image/ directory. The sample input image is named woman_face.jpg.
Prepare the overlay images for the left eye, right eye, and nose in both male and female versions. Save them in the ./image/ directory.
Male versions: right_eye.png, left_eye.png, nose.png
Female versions: right_eye_w.png, left_eye_w.png, nose_w.png

Run the script using the following command:
Copy code
python script_name.py

The output image will be displayed with the facial landmarks and gender-specific overlays.

## Additional Information 
---

The script uses the dlib facial landmark detector and a pre-trained gender classification model from cvlib.
It resizes the output image to a specified width (desired_width variable) for better visualization, maintaining the aspect ratio.

![boy1](https://github.com/cindy-chaewon/opensourcesw_project/assets/144265750/2bd18d57-9656-430f-83cd-aaa952f908db)
![boy2](https://github.com/cindy-chaewon/opensourcesw_project/assets/144265750/8911dd30-43b2-4fd0-a032-90cb854599e3)
![girl1](https://github.com/cindy-chaewon/opensourcesw_project/assets/144265750/42c9459a-8afb-4503-9188-ceda5729fcab)
![girl2](https://github.com/cindy-chaewon/opensourcesw_project/assets/144265750/f494863a-9bbd-44e5-a10a-5a4d71444d95)

## Files 
---
script_name.py: The main Python script for facial landmark detection, gender prediction, and image overlay.
shape_predictor_68_face_landmarks.dat: Pre-trained facial landmark predictor file for dlib. You need to download this file and place it in the same directory as the script. You can find it here.
