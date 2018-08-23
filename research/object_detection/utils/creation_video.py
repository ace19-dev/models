import os
import cv2

IMAGE_DIR = '/home/ace19/training_record/MOT/results/result-20180823/inference_result/MOT17-06/'

v = cv2.VideoWriter(IMAGE_DIR + 'out_video.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (1920, 1080))

num = 0
for path, _, files in os.walk(IMAGE_DIR):
    for file in sorted(files):
        if file.endswith('.jpg'):
            image = cv2.imread(IMAGE_DIR+file)
            v.write(image)
            num += 1
            print(file, ':', num)

v.release()