import os

from lxml import etree

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.core import box_list
from object_detection.core import box_list_ops

tf.logging.set_verbosity(tf.logging.INFO)


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

flags.DEFINE_boolean('ignore_difficult_instances', True, 'Whether to ignore '
                                                  'difficult instances')


FLAGS = flags.FLAGS

#####################################
# read inference infos from out.txt
#####################################
with open(FLAGS.output_path, 'r') as f:
    detected_box_list = f.read().splitlines()


#####################################
# edit inference result for eval
#####################################
detected_results = {}
for detected_box_info in detected_box_list:
    info = detected_box_info.split(',')
    if info[0] in detected_results:
        detected_results[info[0]].append([float(info[1]),float(info[2]),float(info[3]),float(info[4])])
    else:
        detected_results[info[0]] = [[float(info[1]),float(info[2]),float(info[3]),float(info[4])]]


def get_iou(sess, img_name, data):
    # check whether object is detected or not
    try:
        detected_boxes = detected_results[img_name]
    except KeyError:
        return None

    gt_boxes = []
    if 'object' in data:
        for obj in data['object']:
            if obj['name'] != 'excavator':
                continue

            box = [float(obj['bndbox']['ymin']),
                    float(obj['bndbox']['xmin']),
                    float(obj['bndbox']['ymax']),
                    float(obj['bndbox']['xmax'])]
            gt_boxes.append(box)

    boxes1 = box_list.BoxList(tf.constant(gt_boxes))
    boxes2 = box_list.BoxList(tf.constant(detected_boxes))

    # get [N, M] tensor
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
                tf.logging.info('On image %d of %d', idx, len(image_list))

            path = os.path.join(FLAGS.data_dir, FLAGS.annotations_dir, img_name[:-4] + '.xml')

            # check whether anno file exist or not
            try:
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
            except:
                continue

            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            iou_vals = get_iou(sess, img_name, data)
            if iou_vals is not None:
                for ious_per_gt in iou_vals:
                    detected_fail_num = 0
                    for iou_val in ious_per_gt:
                        if iou_val == 0:
                            detected_fail_num += 1

                    if detected_fail_num == len(ious_per_gt):
                        failure += 1
                    else:
                        success += 1

                print('idx:{}, filename:{}, IoU:{}'.format(idx, img_name, iou_vals))
            else:
                failure += 1


        tf.logging.info('success:{}, failure:{}'.format(success, failure))
        tf.logging.info('recall : {}'.format(success / (success + failure)))





