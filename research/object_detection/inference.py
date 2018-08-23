# Object detection inference
#

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image
# from matplotlib import pyplot as plt
import cv2

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
from object_detection.utils import ops as utils_ops

# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# imports from the object detection module.
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# os.environ['CUDA_VISIBLE_DEVICES'] = ''


######################################################################
# Model preparation
#
# Any model exported using the export_inference_graph.py tool can be loaded here simply
# by changing PATH_TO_CKPT to point to a new .pb file.
#
######################################################################
# What model to download.
# MODEL_NAME = 'checkpoints/mask_rcnn_resnet101_atrous_coco_2018_01_28'
# MODEL_NAME = 'checkpoints/excavator_2018-07-30'
MODEL_NAME = 'checkpoints/mot_2018-08-23'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mot_label_map.pbtxt')
NUM_CLASSES = 1


#####################
# Download Model
#####################
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())


####################################################
# Load a (frozen) Tensorflow model into memory
####################################################
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


#######################
# Loading label map
#######################
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


####################
# Helper code
####################
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)



#################
# Detection
#################
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = '/home/ace19/dl_data/KDP/test_images'
subfolder_name = "MOT17-14"
PATH_TO_TEST_IMAGES_DIR = '/home/ace19/dl_data/MOT/MOT17/test/' + subfolder_name + '/img1'
# PATH_TO_TEST_IMAGES_DIR = '/home/ace19/dl_data/MOT/MOT17/test/sample-test'
# PATH_TO_INFERENCE_SAVE_DIR = '/home/ace19/training_record/KDP/records/result-20180803/inference_result'
PATH_TO_INFERENCE_SAVE_DIR = '/home/ace19/training_record/MOT/results/result-20180823/inference_result/' + subfolder_name
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.JPG'.format(i)) for i in range(1, 11) ]
image_names = os.listdir(PATH_TO_TEST_IMAGES_DIR)
image_names.sort()

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, tensor_dict):
    # with graph.as_default():
    #     with tf.Session() as sess:
            # Get handles to input and output tensors
    # ops = tf.get_default_graph().get_operations()
    # all_tensor_names = {output.name for op in ops for output in op.outputs}
    # tensor_dict = {}
    # for key in [
    #     'num_detections', 'detection_boxes', 'detection_scores',
    #     'detection_classes', 'detection_masks'
    # ]:
    #     tensor_name = key + ':0'
    #     if tensor_name in all_tensor_names:
    #         tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    start_time = time.time()

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})
    # output_dict = sess.run(tensor_dict,
    #                        feed_dict={image_tensor: np.expand_dims(image, 0)}, \
    #                        options=options, run_metadata=run_metadata)
    print('Speed %.3f sec' % (time.time() - start_time))

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    # return output_dict, run_metadata
    return output_dict


with detection_graph.as_default():
    with tf.Session() as sess:

        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)


        # In order to save box info of images
        boxes_info_per_images = []
        # plt.figure(figsize=IMAGE_SIZE)
        for image_name in image_names:
            # image = Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR, image_name))
            # # the array based representation of the image will be used later in order to prepare the
            # # result image with boxes and labels on it.
            # image_np = load_image_into_numpy_array(image)

            # start_time = time.time()

            image_np = cv2.imread(os.path.join(PATH_TO_TEST_IMAGES_DIR, image_name))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Actual detection.
            # output_dict, run_metadata = run_inference_for_single_image(image_np, detection_graph)
            output_dict = run_inference_for_single_image(image_np, tensor_dict)

            #####################################
            ### caution : edit depends on biz ###
            #####################################
            # Visualization of the results of a detection.
            _, boxes_info = vis_util.visualize_boxes_and_labels_on_image_array(
                image_name,
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=3)

            # plt.imshow(image_np)
            # plt.show()
            ##############
            # save image
            ##############
            # plt.savefig(PATH_TO_INFERENCE_SAVE_DIR + '/infer_' + image_name, dpi=100)
            # plt.savefig(PATH_TO_INFERENCE_SAVE_DIR + '/infer_' + image_name)

            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(PATH_TO_INFERENCE_SAVE_DIR + '/infer_' + image_name, image_np)

            boxes_info_per_images.extend(boxes_info)

            # print("image_name: ", image_name)
            # print('Time %.3f sec' % (time.time() - start_time))

        # write out boxes_info to file
        with open(os.path.join(PATH_TO_INFERENCE_SAVE_DIR, subfolder_name + '.txt'), 'a') as f:
        # with open(os.path.join(PATH_TO_INFERENCE_SAVE_DIR, 'out_file.txt'), 'a') as f:
            for info in boxes_info_per_images:
                f.write(info + '\n')



# To disable GPU, add below code
# tf.where and other post-processing operations are running anomaly slow on GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# fetched_timeline = timeline.Timeline(run_metadata.step_stats)
# chrome_trace = fetched_timeline.generate_chrome_trace_format()
# with open('Experiment_1.json', 'w') as f:
#     f.write(chrome_trace)