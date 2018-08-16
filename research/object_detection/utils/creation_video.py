import os
import cv2

IMAGE_DIR = '/home/ace19/training_record/MOT/records/result_det_file-20180816/MOT17-03/'

v = cv2.VideoWriter(IMAGE_DIR + 'output.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (1920, 1080))

for path, _, files in os.walk(IMAGE_DIR):
    for file in sorted(files):
        if file.endswith('.jpg'):
            image = cv2.imread(IMAGE_DIR+file)
            v.write(image)
            print(file)

v.release()