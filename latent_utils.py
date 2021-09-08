import random

import keras.backend as K
import numpy as np
import tensorflow as tf

random.seed(8888)


def fix_gpu_memory():
    """Fixs GPU memory.

    # Args:

    # Returns:
      TF session.
    """
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    # init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess


def construct_mask(num_patterns=1, image_dim=32, channel_num=3,
                   pattern_size=6, randomize=False):
    """Construct mask of trigger in a hard-coded way.

    # Args:
      num_pattern: number of patterns.
      image_dim: dimension of input image.
      channel_num: number of image channels.
      pattern_size: size of trigger with unit in number of pixels.
      randomize: True if trigger color is randomized. Else, it uses white.

    # Returns:
      A tuple of (mask, pattern).
    """
    mask = np.zeros((image_dim, image_dim, channel_num))
    pattern = np.zeros((image_dim, image_dim, channel_num))
    r = list(range(0, num_patterns + 1))
    res = []
    for row in range(image_dim):
        for col in range(image_dim):
            # Bottom right
            if (1 in r) and \
                    row > image_dim - 2 - pattern_size and \
                    row < image_dim - 1 and \
                    col > image_dim - 2 - pattern_size and \
                    col < image_dim - 1:
                for c in range(channel_num):
                    if randomize:
                        select_color = 255.0 * random.uniform(0, 1)
                        pattern[row][col][c] = select_color
                        mask[row][col][c] = 1
                        res.append((row, col, select_color))
                    else:
                        pattern[row][col][c] = 255.0
                        mask[row][col][c] = 1
                        res.append((row, col, 1.0))
            # Top right
            if (2 in r) and \
                    row > 0 and \
                    row < pattern_size + 1 and \
                    col > image_dim - 2 - pattern_size and \
                    col < image_dim - 1:
                for c in range(channel_num):
                    if randomize:
                        select_color = 255.0 * random.uniform(0, 1)
                        pattern[row][col][c] = select_color
                        mask[row][col][c] = 1
                        res.append((row, col, select_color))
                    else:
                        pattern[row][col][c] = 255.0
                        mask[row][col][c] = 1
                        res.append((row, col, 1.0))

            # Bottom Left
            if (3 in r) and \
                    col > 0 and \
                    col < pattern_size + 1 and \
                    row > image_dim - 2 - pattern_size and \
                    row < image_dim - 1:
                for c in range(channel_num):
                    if randomize:
                        select_color = 255.0 * random.uniform(0, 1)
                        pattern[row][col][c] = select_color
                        mask[row][col][c] = 1
                        res.append((row, col, select_color))
                    else:
                        pattern[row][col][c] = 255.0
                        mask[row][col][c] = 1
                        res.append((row, col, 1.0))
            # Top left
            if (4 in r) and \
                    col > 0 and \
                    col < pattern_size + 1 and \
                    row > 0 and \
                    row < pattern_size + 1:
                for c in range(channel_num):
                    if randomize:
                        select_color = 255.0 * random.uniform(0, 1)
                        pattern[row][col][c] = select_color
                        mask[row][col][c] = 1
                        res.append((row, col, select_color))
                    else:
                        pattern[row][col][c] = 255.0
                        mask[row][col][c] = 1
                        res.append((row, col, 1.0))
            # Top center
            if (5 in r) and \
                    col > image_dim // 2 - 3 and \
                    col < image_dim // 2 + 2 and \
                    row > 0 and \
                    row < 5:
                for c in range(channel_num):
                    if randomize:
                        select_color = 255.0 * random.uniform(0, 1)
                        pattern[row][col][c] = select_color
                        mask[row][col][c] = 1
                        res.append((row, col, select_color))
                    else:
                        pattern[row][col][c] = 255.0
                        mask[row][col][c] = 1
                        res.append((row, col, 1.0))
            # center right
            if (6 in r) and \
                    row > image_dim // 2 - 3 and \
                    row < image_dim // 2 + 2 and \
                    col > 0 and \
                    col < 5:
                for c in range(channel_num):
                    if randomize:
                        select_color = 255.0 * random.uniform(0, 1)
                        pattern[row][col][c] = select_color
                        mask[row][col][c] = 1
                        res.append((row, col, select_color))
                    else:
                        pattern[row][col][c] = 255.0
                        mask[row][col][c] = 1
                        res.append((row, col, 1.0))

            # center left
            if (7 in r) and \
                    row > image_dim // 2 - 3 and \
                    row < image_dim // 2 + 2 and \
                    col > image_dim - 2 - pattern_size and \
                    col < image_dim - 1:
                for c in range(channel_num):
                    if randomize:
                        select_color = 255.0 * random.uniform(0, 1)
                        pattern[row][col][c] = select_color
                        mask[row][col][c] = 1
                        res.append((row, col, select_color))
                    else:
                        pattern[row][col][c] = 255.0
                        mask[row][col][c] = 1
                        res.append((row, col, 1.0))

            # bottom center
            if (8 in r) and \
                    col > image_dim // 2 - 3 and \
                    col < image_dim // 2 + 2 and \
                    row > image_dim - 2 - pattern_size and \
                    row < image_dim - 1:
                for c in range(channel_num):
                    if randomize:
                        select_color = 255.0 * random.uniform(0, 1)
                        pattern[row][col][c] = select_color
                        mask[row][col][c] = 1
                        res.append((row, col, select_color))
                    else:
                        pattern[row][col][c] = 255.0
                        mask[row][col][c] = 1
                        res.append((row, col, 1.0))

            # center center
            if (9 in r) and \
                    col > image_dim // 2 - 3 and \
                    col < image_dim // 2 + 2 and \
                    row > image_dim // 2 - 3 and \
                    row < image_dim // 2 + 2:
                for c in range(channel_num):
                    if randomize:
                        select_color = 255.0 * random.uniform(0, 1)
                        pattern[row][col][c] = select_color
                        mask[row][col][c] = 1
                        res.append((row, col, select_color))
                    else:
                        pattern[row][col][c] = 255.0
                        mask[row][col][c] = 1
                        res.append((row, col, 1.0))

    return mask, pattern


def imagenet_reverse_preprocessing(x, data_format=None):
    """Reverse-preprocesses a tensor encoding a batch of images.

    # Args:
      x: input Numpy tensor of a batch of preproccessed images in 4D.
      data_format: data format of the image tensor.

    # Returns:
      Image tensor before preprocessing.
    """
    x = np.array(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    if data_format == 'channels_first':
        # Zero-center by mean pixel
        x[:, 0, :, :] += 103.939
        x[:, 1, :, :] += 116.779
        x[:, 2, :, :] += 123.68
        # 'BGR'->'RGB'
        x = x[:, ::-1, :, :]
    else:
        # Zero-center by mean pixel
        x[:, :, :, 0] += 103.939
        x[:, :, :, 1] += 116.779
        x[:, :, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, :, ::-1]
    return x
