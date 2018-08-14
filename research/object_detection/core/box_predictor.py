# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Box predictor for object detectors.

Box predictors are classes that take a high level
image feature map as input and produce two predictions,
(1) a tensor encoding box locations, and
(2) a tensor encoding classes for each box.

These components are passed directly to loss functions
in our detection models.

These modules are separated from the main model since the same
few box predictor architectures are shared across many models.
"""
from abc import abstractmethod
import tensorflow as tf

BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'
MASK_PREDICTIONS = 'mask_predictions'


class BoxPredictor(object):
  """BoxPredictor."""

  def __init__(self, is_training, num_classes):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
    """
    self._is_training = is_training
    self._num_classes = num_classes

  @property
  def is_keras_model(self):
    return False

  @property
  def num_classes(self):
    return self._num_classes

  def predict(self, image_features, num_predictions_per_location,
              scope=None, **params):
    """Computes encoded object locations and corresponding confidences.

    Takes a list of high level image feature maps as input and produces a list
    of box encodings and a list of class scores where each element in the output
    lists correspond to the feature maps in the input list.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
      width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
      scope: Variable and Op scope name.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors of shape
          [batch_size, num_anchors_i, q, code_size] representing the location of
          the objects, where q is 1 or the number of classes. Each entry in the
          list corresponds to a feature map in the input `image_features` list.
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.

    Raises:
      ValueError: If length of `image_features` is not equal to length of
        `num_predictions_per_location`.
    """
    if len(image_features) != len(num_predictions_per_location):
      raise ValueError('image_feature and num_predictions_per_location must '
                       'be of same length, found: {} vs {}'.
                       format(len(image_features),
                              len(num_predictions_per_location)))
    if scope is not None:
      with tf.variable_scope(scope):
        return self._predict(image_features, num_predictions_per_location,
                             **params)
    return self._predict(image_features, num_predictions_per_location,
                         **params)

  # TODO(rathodv): num_predictions_per_location could be moved to constructor.
  # This is currently only used by ConvolutionalBoxPredictor.
  @abstractmethod
  def _predict(self, image_features, num_predictions_per_location, **params):
    """Implementations must override this method.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors of shape
          [batch_size, num_anchors_i, q, code_size] representing the location of
          the objects, where q is 1 or the number of classes. Each entry in the
          list corresponds to a feature map in the input `image_features` list.
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.
    """
    pass


class KerasBoxPredictor(tf.keras.Model):
  """Keras-based BoxPredictor."""

  def __init__(self, is_training, num_classes, freeze_batchnorm,
               inplace_batchnorm_update):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
    """
    super(KerasBoxPredictor, self).__init__()

    self._is_training = is_training
    self._num_classes = num_classes
    self._freeze_batchnorm = freeze_batchnorm
    self._inplace_batchnorm_update = inplace_batchnorm_update

  @property
  def is_keras_model(self):
    return True

  @property
  def num_classes(self):
    return self._num_classes

  def call(self, image_features, scope=None, **kwargs):
    """Computes encoded object locations and corresponding confidences.

    Takes a list of high level image feature maps as input and produces a list
    of box encodings and a list of class scores where each element in the output
    lists correspond to the feature maps in the input list.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
      width_i, channels_i] containing features for a batch of images.
      scope: Variable and Op scope name.
      **kwargs: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors. Each entry in the list
          corresponds to a feature map in the input `image_features` list. All
          tensors in the list have one of the two following shapes:
          a. [batch_size, num_anchors_i, q, code_size] representing the location
            of the objects, where q is 1 or the number of classes.
          b. [batch_size, num_anchors_i, code_size].
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.
    """
    return self._predict(image_features, **kwargs)

  @abstractmethod
  def _predict(self, image_features, **kwargs):
    """Implementations must override this method.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
<<<<<<< HEAD
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map.

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.
    """
    box_encodings_list = []
    class_predictions_list = []
    # TODO(rathodv): Come up with a better way to generate scope names
    # in box predictor once we have time to retrain all models in the zoo.
    # The following lines create scope names to be backwards compatible with the
    # existing checkpoints.
    box_predictor_scopes = [_NoopVariableScope()]
    if len(image_features) > 1:
      box_predictor_scopes = [
          tf.variable_scope('BoxPredictor_{}'.format(i))
          for i in range(len(image_features))
      ]

    for (image_feature,
         num_predictions_per_location, box_predictor_scope) in zip(
             image_features, num_predictions_per_location_list,
             box_predictor_scopes):
      with box_predictor_scope:
        # Add a slot for the background class.
        num_class_slots = self.num_classes + 1
        net = image_feature
        with slim.arg_scope(self._conv_hyperparams_fn()), \
             slim.arg_scope([slim.dropout], is_training=self._is_training):
          # Add additional conv layers before the class predictor.
          features_depth = static_shape.get_depth(image_feature.get_shape())
          depth = max(min(features_depth, self._max_depth), self._min_depth)
          tf.logging.info('depth of additional conv before box predictor: {}'.
                          format(depth))
          if depth > 0 and self._num_layers_before_predictor > 0:
            for i in range(self._num_layers_before_predictor):
              net = slim.conv2d(
                  net, depth, [1, 1], scope='Conv2d_%d_1x1_%d' % (i, depth))
          with slim.arg_scope([slim.conv2d], activation_fn=None,
                              normalizer_fn=None, normalizer_params=None):
            if self._use_depthwise:
              box_encodings = slim.separable_conv2d(
                  net, None, [self._kernel_size, self._kernel_size],
                  padding='SAME', depth_multiplier=1, stride=1,
                  rate=1, scope='BoxEncodingPredictor_depthwise')
              box_encodings = slim.conv2d(
                  box_encodings,
                  num_predictions_per_location * self._box_code_size, [1, 1],
                  scope='BoxEncodingPredictor')
            else:
              box_encodings = slim.conv2d(
                  net, num_predictions_per_location * self._box_code_size,
                  [self._kernel_size, self._kernel_size],
                  scope='BoxEncodingPredictor')
            if self._use_dropout:
              net = slim.dropout(net, keep_prob=self._dropout_keep_prob)
            if self._use_depthwise:
              class_predictions_with_background = slim.separable_conv2d(
                  net, None, [self._kernel_size, self._kernel_size],
                  padding='SAME', depth_multiplier=1, stride=1,
                  rate=1, scope='ClassPredictor_depthwise')
              class_predictions_with_background = slim.conv2d(
                  class_predictions_with_background,
                  num_predictions_per_location * num_class_slots,
                  [1, 1], scope='ClassPredictor')
            else:
              class_predictions_with_background = slim.conv2d(
                  net, num_predictions_per_location * num_class_slots,
                  [self._kernel_size, self._kernel_size],
                  scope='ClassPredictor',
                  biases_initializer=tf.constant_initializer(
                      self._class_prediction_bias_init))
            if self._apply_sigmoid_to_scores:
              class_predictions_with_background = tf.sigmoid(
                  class_predictions_with_background)

        combined_feature_map_shape = (shape_utils.
                                      combined_static_and_dynamic_shape(
                                          image_feature))
        box_encodings = tf.reshape(
            box_encodings, tf.stack([combined_feature_map_shape[0],
                                     combined_feature_map_shape[1] *
                                     combined_feature_map_shape[2] *
                                     num_predictions_per_location,
                                     1, self._box_code_size]))
        box_encodings_list.append(box_encodings)
        class_predictions_with_background = tf.reshape(
            class_predictions_with_background,
            tf.stack([combined_feature_map_shape[0],
                      combined_feature_map_shape[1] *
                      combined_feature_map_shape[2] *
                      num_predictions_per_location,
                      num_class_slots]))
        class_predictions_list.append(class_predictions_with_background)
    return {
        BOX_ENCODINGS: box_encodings_list,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list
    }


# TODO(rathodv): Replace with slim.arg_scope_func_key once its available
# externally.
def _arg_scope_func_key(op):
  """Returns a key that can be used to index arg_scope dictionary."""
  return getattr(op, '_key_op', str(op))


# TODO(rathodv): Merge the implementation with ConvolutionalBoxPredictor above
# since they are very similar.
class WeightSharedConvolutionalBoxPredictor(BoxPredictor):
  """Convolutional Box Predictor with weight sharing.

  Defines the box predictor as defined in
  https://arxiv.org/abs/1708.02002. This class differs from
  ConvolutionalBoxPredictor in that it shares weights and biases while
  predicting from different feature maps.  Separate multi-layer towers are
  constructed for the box encoding and class predictors respectively.
  """

  def __init__(self,
               is_training,
               num_classes,
               conv_hyperparams_fn,
               depth,
               num_layers_before_predictor,
               box_code_size,
               kernel_size=3,
               class_prediction_bias_init=0.0,
               use_dropout=False,
               dropout_keep_prob=0.8,
               share_prediction_tower=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      depth: depth of conv layers.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      box_code_size: Size of encoding for each box.
      kernel_size: Size of final convolution kernel.
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.
      use_dropout: Whether to apply dropout to class prediction head.
      dropout_keep_prob: Probability of keeping activiations.
      share_prediction_tower: Whether to share the multi-layer tower between box
        prediction and class prediction heads.
    """
    super(WeightSharedConvolutionalBoxPredictor, self).__init__(is_training,
                                                                num_classes)
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._depth = depth
    self._num_layers_before_predictor = num_layers_before_predictor
    self._box_code_size = box_code_size
    self._kernel_size = kernel_size
    self._class_prediction_bias_init = class_prediction_bias_init
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._share_prediction_tower = share_prediction_tower

  def _predict(self, image_features, num_predictions_per_location_list):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels] containing features for a batch of images. Note that
        when not all tensors in the list have the same number of channels, an
        additional projection layer will be added on top the tensor to generate
        feature map with number of channels consitent with the majority.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map. Note that all values must be the same since the weights are
        shared.

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.

    Raises:
      ValueError: If the image feature maps do not have the same number of
        channels or if the num predictions per locations is differs between the
        feature maps.
    """
    if len(set(num_predictions_per_location_list)) > 1:
      raise ValueError('num predictions per location must be same for all'
                       'feature maps, found: {}'.format(
                           num_predictions_per_location_list))
    feature_channels = [
        image_feature.shape[3].value for image_feature in image_features
    ]
    has_different_feature_channels = len(set(feature_channels)) > 1
    if has_different_feature_channels:
      inserted_layer_counter = 0
      target_channel = max(set(feature_channels), key=feature_channels.count)
      tf.logging.info('Not all feature maps have the same number of '
                      'channels, found: {}, addition project layers '
                      'to bring all feature maps to uniform channels '
                      'of {}'.format(feature_channels, target_channel))
    box_encodings_list = []
    class_predictions_list = []
    num_class_slots = self.num_classes + 1
    for feature_index, (image_feature,
                        num_predictions_per_location) in enumerate(
                            zip(image_features,
                                num_predictions_per_location_list)):
      # Add a slot for the background class.
      with tf.variable_scope('WeightSharedConvolutionalBoxPredictor',
                             reuse=tf.AUTO_REUSE):
        with slim.arg_scope(self._conv_hyperparams_fn()) as sc:
          apply_batch_norm = _arg_scope_func_key(slim.batch_norm) in sc
          # Insert an additional projection layer if necessary.
          if (has_different_feature_channels and
              image_feature.shape[3].value != target_channel):
            image_feature = slim.conv2d(
                image_feature,
                target_channel, [1, 1],
                stride=1,
                padding='SAME',
                activation_fn=None,
                normalizer_fn=(tf.identity if apply_batch_norm else None),
                scope='ProjectionLayer/conv2d_{}'.format(
                    inserted_layer_counter))
            if apply_batch_norm:
              image_feature = slim.batch_norm(
                  image_feature,
                  scope='ProjectionLayer/conv2d_{}/BatchNorm'.format(
                      inserted_layer_counter))
            inserted_layer_counter += 1
          box_encodings_net = image_feature
          class_predictions_net = image_feature
          for i in range(self._num_layers_before_predictor):
            box_prediction_tower_prefix = (
                'PredictionTower' if self._share_prediction_tower
                else 'BoxPredictionTower')
            box_encodings_net = slim.conv2d(
                box_encodings_net,
                self._depth,
                [self._kernel_size, self._kernel_size],
                stride=1,
                padding='SAME',
                activation_fn=None,
                normalizer_fn=(tf.identity if apply_batch_norm else None),
                scope='{}/conv2d_{}'.format(box_prediction_tower_prefix, i))
            if apply_batch_norm:
              box_encodings_net = slim.batch_norm(
                  box_encodings_net,
                  scope='{}/conv2d_{}/BatchNorm/feature_{}'.
                  format(box_prediction_tower_prefix, i, feature_index))
            box_encodings_net = tf.nn.relu6(box_encodings_net)
          box_encodings = slim.conv2d(
              box_encodings_net,
              num_predictions_per_location * self._box_code_size,
              [self._kernel_size, self._kernel_size],
              activation_fn=None, stride=1, padding='SAME',
              normalizer_fn=None,
              scope='BoxEncodingPredictor')

          if self._share_prediction_tower:
            class_predictions_net = box_encodings_net
          else:
            for i in range(self._num_layers_before_predictor):
              class_predictions_net = slim.conv2d(
                  class_predictions_net,
                  self._depth,
                  [self._kernel_size, self._kernel_size],
                  stride=1,
                  padding='SAME',
                  activation_fn=None,
                  normalizer_fn=(tf.identity if apply_batch_norm else None),
                  scope='ClassPredictionTower/conv2d_{}'.format(i))
              if apply_batch_norm:
                class_predictions_net = slim.batch_norm(
                    class_predictions_net,
                    scope='ClassPredictionTower/conv2d_{}/BatchNorm/feature_{}'
                    .format(i, feature_index))
              class_predictions_net = tf.nn.relu6(class_predictions_net)
          if self._use_dropout:
            class_predictions_net = slim.dropout(
                class_predictions_net, keep_prob=self._dropout_keep_prob)
          class_predictions_with_background = slim.conv2d(
              class_predictions_net,
              num_predictions_per_location * num_class_slots,
              [self._kernel_size, self._kernel_size],
              activation_fn=None, stride=1, padding='SAME',
              normalizer_fn=None,
              biases_initializer=tf.constant_initializer(
                  self._class_prediction_bias_init),
              scope='ClassPredictor')

          combined_feature_map_shape = (shape_utils.
                                        combined_static_and_dynamic_shape(
                                            image_feature))
          box_encodings = tf.reshape(
              box_encodings, tf.stack([combined_feature_map_shape[0],
                                       combined_feature_map_shape[1] *
                                       combined_feature_map_shape[2] *
                                       num_predictions_per_location,
                                       1, self._box_code_size]))
          box_encodings_list.append(box_encodings)
          class_predictions_with_background = tf.reshape(
              class_predictions_with_background,
              tf.stack([combined_feature_map_shape[0],
                        combined_feature_map_shape[1] *
                        combined_feature_map_shape[2] *
                        num_predictions_per_location,
                        num_class_slots]))
          class_predictions_list.append(class_predictions_with_background)
    return {
        BOX_ENCODINGS: box_encodings_list,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list
    }
=======
      **kwargs: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors. Each entry in the list
          corresponds to a feature map in the input `image_features` list. All
          tensors in the list have one of the two following shapes:
          a. [batch_size, num_anchors_i, q, code_size] representing the location
            of the objects, where q is 1 or the number of classes.
          b. [batch_size, num_anchors_i, code_size].
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.
    """
    raise NotImplementedError
>>>>>>> upstream/master
