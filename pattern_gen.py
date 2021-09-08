from decimal import Decimal

import numpy as np
from keras import backend as K
from keras.layers import Cropping2D, UpSampling2D
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.utils import to_categorical


class PatternGenerator:

    # upsample size, default is 1
    UPSAMPLE_SIZE = 1
    # pixel intensity range of image and preprocessing method
    # raw: [0, 255]
    # mnist: [0, 1]
    # imagenet: imagenet mean centering
    # inception: [-1, 1]
    INTENSITY_RANGE = 'raw'
    # patience
    PATIENCE = 10
    # multiple of changing cost, down multiple is the square root of this
    COST_MULTIPLIER = 1.5,
    # if resetting cost to 0 at the beginning
    # default is true for full optimization, set to false for early detection
    RESET_COST_TO_ZERO = True
    # min/max of mask
    MASK_MIN = 0
    MASK_MAX = 1
    # min/max of raw pixel intensity
    COLOR_MIN = 0
    COLOR_MAX = 255
    # number of color channel
    IMG_COLOR = 3
    # whether to shuffle during each epoch
    SHUFFLE = True
    # batch size of optimization
    BATCH_SIZE = 32
    # verbose level, 0, 1 or 2
    VERBOSE = 1
    # whether to return log or not
    RETURN_LOGS = True
    # whether to save last pattern or best pattern
    SAVE_LAST = False
    # epsilon used in tanh
    EPSILON = K.epsilon()
    # early stop flag
    EARLY_STOP = True
    # early stop threshold
    EARLY_STOP_THRESHOLD = 0.99
    # early stop patience
    EARLY_STOP_PATIENCE = 2 * PATIENCE
    # save tmp masks, for debugging purpose
    SAVE_TMP = False
    # dir to save intermediate masks
    TMP_DIR = 'tmp'
    # whether input image has been preprocessed or not
    RAW_INPUT_FLAG = False

    def __init__(self, bottleneck_model, mask, intensity_range, input_shape,
                 init_cost, steps, mini_batch, lr,
                 upsample_size=UPSAMPLE_SIZE,
                 patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
                 reset_cost_to_zero=RESET_COST_TO_ZERO,
                 mask_min=MASK_MIN, mask_max=MASK_MAX,
                 color_min=COLOR_MIN, color_max=COLOR_MAX, img_color=IMG_COLOR,
                 shuffle=SHUFFLE, batch_size=BATCH_SIZE, verbose=VERBOSE,
                 return_logs=RETURN_LOGS, save_last=SAVE_LAST,
                 epsilon=EPSILON,
                 early_stop=EARLY_STOP,
                 early_stop_threshold=EARLY_STOP_THRESHOLD,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 save_tmp=SAVE_TMP, tmp_dir=TMP_DIR,
                 raw_input_flag=RAW_INPUT_FLAG):

        assert intensity_range in {'imagenet', 'inception', 'mnist', 'raw'}

        self.bottleneck_model = bottleneck_model
        self.mask = mask
        self.intensity_range = intensity_range
        self.input_shape = input_shape
        self.init_cost = init_cost
        self.steps = steps
        self.mini_batch = mini_batch
        self.lr = lr
        self.upsample_size = upsample_size
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.reset_cost_to_zero = reset_cost_to_zero
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.color_min = color_min
        self.color_max = color_max
        self.img_color = img_color
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.save_tmp = save_tmp
        self.tmp_dir = tmp_dir
        self.raw_input_flag = raw_input_flag

        pattern = np.zeros(input_shape)
        pattern_tanh = np.zeros_like(pattern)

        # prepare mask related tensors
        mask_tensor_unrepeat = K.variable(mask)
        self.mask_tensor = K.expand_dims(mask_tensor_unrepeat, axis=0)
        upsample_layer = UpSampling2D(
            size=(self.upsample_size, self.upsample_size))
        mask_upsample_tensor_uncrop = upsample_layer(self.mask_tensor)
        uncrop_shape = K.int_shape(mask_upsample_tensor_uncrop)[1:]
        cropping_layer = Cropping2D(
            cropping=((0, uncrop_shape[0] - self.input_shape[0]),
                      (0, uncrop_shape[1] - self.input_shape[1])))
        self.mask_upsample_tensor = cropping_layer(
            mask_upsample_tensor_uncrop)
        reverse_mask_tensor = (K.ones_like(self.mask_upsample_tensor) -
                               self.mask_upsample_tensor)

        def keras_preprocess(x_input, intensity_range):

            if intensity_range is 'raw':
                x_preprocess = x_input

            elif intensity_range is 'imagenet':
                # 'RGB'->'BGR'
                x_tmp = x_input[..., ::-1]
                # Zero-center by mean pixel
                mean = K.constant([[[103.939, 116.779, 123.68]]])
                x_preprocess = x_tmp - mean

            elif intensity_range is 'inception':
                x_preprocess = (x_input / 255.0 - 0.5) * 2.0

            elif intensity_range is 'mnist':
                x_preprocess = x_input / 255.0

            else:
                raise Exception('unknown intensity_range %s' % intensity_range)

            return x_preprocess

        def keras_reverse_preprocess(x_input, intensity_range):

            if intensity_range is 'raw':
                x_reverse = x_input

            elif intensity_range is 'imagenet':
                # Zero-center by mean pixel
                mean = K.constant([[[103.939, 116.779, 123.68]]])
                x_reverse = x_input + mean
                # 'BGR'->'RGB'
                x_reverse = x_reverse[..., ::-1]

            elif intensity_range is 'inception':
                x_reverse = (x_input / 2 + 0.5) * 255.0

            elif intensity_range is 'mnist':
                x_reverse = x_input * 255.0

            else:
                raise Exception('unknown intensity_range %s' % intensity_range)

            return x_reverse

        # prepare pattern related tensors
        self.pattern_tanh_tensor = K.variable(pattern_tanh)
        self.pattern_raw_tensor = (
            (K.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5) *
            255.0)

        # prepare input image related tensors
        # ignore clip operation here
        # assume input image is already clipped into valid color range
        input_tensor = K.placeholder(bottleneck_model.input_shape)
        if self.raw_input_flag:
            input_raw_tensor = input_tensor
        else:
            input_raw_tensor = keras_reverse_preprocess(
                input_tensor, self.intensity_range)

        # IMPORTANT: MASK OPERATION IN RAW DOMAIN
        X_adv_raw_tensor = (
            reverse_mask_tensor * input_raw_tensor +
            self.mask_upsample_tensor * self.pattern_raw_tensor)

        X_adv_tensor = keras_preprocess(X_adv_raw_tensor, self.intensity_range)

        output_tensor = bottleneck_model(X_adv_tensor)
        y_true_tensor = K.placeholder(bottleneck_model.output_shape)

        def sub_mean_squared_error(y_true, y_pred):
            return K.mean(K.square(y_pred[:, :100] - y_true[:, :100]), axis=-1)

        self.loss_mse = sub_mean_squared_error(output_tensor, y_true_tensor)

        cost = self.init_cost
        self.cost_tensor = K.variable(cost)
        self.loss = self.loss_mse

        self.opt = Adam(lr=self.lr, beta_1=0.5, beta_2=0.9)
        self.updates = self.opt.get_updates(
            params=[self.pattern_tanh_tensor],
            loss=self.loss_mse)
        self.train = K.function(
            [input_tensor, y_true_tensor],
            [self.loss_mse],
            updates=self.updates)

        pass

    def reset_opt(self):

        K.set_value(self.opt.iterations, 0)
        for w in self.opt.weights:
            K.set_value(w, np.zeros(K.int_shape(w)))

        pass

    def reset_state(self, pattern_init):

        print('resetting state')

        # setting cost
        if self.reset_cost_to_zero:
            self.cost = 0
        else:
            self.cost = self.init_cost
        K.set_value(self.cost_tensor, self.cost)

        # setting mask and pattern
        pattern = np.array(pattern_init)
        pattern = np.clip(pattern, self.color_min, self.color_max)

        # convert to tanh space
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - self.epsilon))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        K.set_value(self.pattern_tanh_tensor, pattern_tanh)

        # resetting optimizer states
        self.reset_opt()

        pass

    def generate(self, gen, pattern_init):

        # since we use a single optimizer repeatedly, we need to reset
        # optimzier's internal states before running the optimization
        self.reset_state(pattern_init)

        # best optimization results
        pattern_best = None
        reg_best = float('inf')

        # logs and counters for adjusting balance cost
        logs = []
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        # loop start
        for step in range(self.steps):

            # record loss for all mini-batches
            loss_mse_list = []
            for idx in range(self.mini_batch):
                X_batch, tgt_batch = next(gen)
                loss_mse = self.train([X_batch, tgt_batch])
                loss_mse_list.extend(list(loss_mse))

            avg_loss_mse = np.mean(loss_mse_list)

            # if step % 10 == 0:
            #     self.reset_opt()

            # verbose
            if self.verbose != 0:
                if self.verbose == 2 or step % (self.steps // 10) == 0:
                    print('step: %3d, cost: %.2E, MSE: %f' %
                          (step, Decimal(self.cost), avg_loss_mse))

            # save log
            logs.append((step, avg_loss_mse, self.cost))

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and
                        cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
                    print('early stop')
                    break

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    print('up cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost * self.cost_multiplier_up)))
                self.cost *= self.cost_multiplier_up
                K.set_value(self.cost_tensor, self.cost)
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost / self.cost_multiplier_down)))
                self.cost /= self.cost_multiplier_down
                K.set_value(self.cost_tensor, self.cost)
                cost_down_flag = True

        # save the final version
        mask_best = K.eval(self.mask_tensor)
        mask_best = mask_best[0, ..., 0]
        mask_upsample_best = K.eval(self.mask_upsample_tensor)
        mask_upsample_best = mask_upsample_best[0, ..., 0]
        pattern_best = K.eval(self.pattern_raw_tensor)

        if self.return_logs:
            return pattern_best, mask_best, mask_upsample_best, logs
        else:
            return pattern_best, mask_best, mask_upsample_best
