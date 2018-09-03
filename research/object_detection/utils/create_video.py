import os
import cv2

IMAGE_DIR = '/home/ace19/training_record/MOT/results/frcnn_result-20180827/inference_result_threshold_0.70/MOT17-06/'

v = cv2.VideoWriter(IMAGE_DIR + 'out_video.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (1920, 1080))
# v = cv2.VideoWriter(IMAGE_DIR + 'out_video.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (640, 480))

num = 0
for path, _, files in os.walk(IMAGE_DIR):
    for file in sorted(files):
        if file.endswith('.jpg'):
            image = cv2.imread(IMAGE_DIR+file)
            v.write(image)
            num += 1
            print(file, ':', num)

v.release()