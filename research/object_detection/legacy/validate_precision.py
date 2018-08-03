import os
import logging

from lxml import etree
import numpy as np

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.core import box_list
from object_detection.core import box_list_ops




flags = tf.app.flags
flags.DEFINE_string('data_dir',
                    '/home/ace19/dl_data/excavator2/test',
                    'Root directory to raw PASCAL VOC dataset.')

flags.DEFINE_string('annotations_dir', 'annotations',
                    '(Relative) path to annotations directory.')

flags.DEFINE_string('images_dir', 'images',
                    '(Relative) path to images directory.')

flags.DEFINE_string('output_path',
                    '/home/ace19/training_record/excavator/records/result-20180802/inference_result/out.txt',
                    'Path to output TFRecord')

flags.DEFINE_string('label_map_path',
                    '../data/pascal_label_map.pbtxt',
                    'Path to label map protos')

# flags.DEFINE_boolean('ignore_difficult_instances', True, 'Whether to ignore '
#                                                   'difficult instances')



FLAGS = flags.FLAGS


# read inference infos from out.txt
with open(FLAGS.output_path, 'r') as f:
    ilist = f.read().splitlines()

# edit inference result for eval
infer_results = {}
# for infer in ilist:
#     info = infer.split(',')
#     if info[0] in infer_results:
#         infer_results[info[0]].append(info[1:])
#     else:
#         infer_results[info[0]] = [info[1:]]
for infer in ilist:
    info = infer.split(',')
    if info[0] in infer_results:
        infer_results[info[0]].append([float(info[1]),float(info[2]),float(info[3]),float(info[4])])
    else:
        infer_results[info[0]] = [[float(info[1]),float(info[2]),float(info[3]),float(info[4])]]


def validate(sess, img_name, data):
    try:
        infer = infer_results[img_name]
    except KeyError:
        return None

    boxes = []
    if 'object' in data:
        for obj in data['object']:
            if obj['name'] != 'excavator':
                continue

            box = [float(obj['bndbox']['ymin']),
                    float(obj['bndbox']['xmin']),
                    float(obj['bndbox']['ymax']),
                    float(obj['bndbox']['xmax'])]
            boxes.append(box)


    gt_boxes = tf.constant(boxes)
    infer_boxes = tf.constant(infer)

    boxes1 = box_list.BoxList(gt_boxes)
    boxes2 = box_list.BoxList(infer_boxes)

    iou = box_list_ops.iou(boxes1, boxes2)

    return sess.run(iou)


with tf.Graph().as_default():
    with tf.Session() as sess:
        image_path = os.path.join(FLAGS.data_dir, FLAGS.images_dir)
        image_list = os.listdir(image_path)
        image_list.sort()

        success = 0
        failure = 0
        for idx, img_name in enumerate(image_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(image_list))

            path = os.path.join(FLAGS.data_dir, FLAGS.annotations_dir, img_name[:-4] + '.xml')
            try:
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
            except:
                continue

            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            ious = validate(sess, img_name, data)
            if ious is not None:
                for gt in ious:
                    infer_num = 0
                    for val_iou in gt:
                        if val_iou == 0:
                            infer_num += 1
                    if infer_num == len(gt):
                        failure += 1
                    else:
                        success += 1
                print('idx:{}, filename:{}, IoU:{}'.format(idx, img_name, ious))
            else:
                failure += 1

        print('success:{}, failure:{}'.format(success, failure))





