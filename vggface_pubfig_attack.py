import os
import random

import h5py
import keras
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from tensorflow.python.platform import flags

import latent_utils
from pattern_gen import PatternGenerator

random.seed(8888)
np.random.seed(8888)

FLAGS = flags.FLAGS

# Integer flags.
flags.DEFINE_integer('gpu', 0, 'Which gpu will be used.')
flags.DEFINE_integer('img_width', 224, 'Input image size.')
flags.DEFINE_integer('img_height', 224, 'Input image size.')
flags.DEFINE_integer('nb_channels', 3, 'Input image channel number.')
flags.DEFINE_integer('target_y_tea', 30, 'Target label in Teacher dataset.')
flags.DEFINE_integer('target_y_stu', 17,
                     'Target label in Student dataset. For evaluation only.')
flags.DEFINE_integer('teacher_retrain_epochs', 10,
                     'Number of epochs of Teacher retraining.')
flags.DEFINE_integer('inject_batch_size', 16,
                     'Batch size used in backdoor injection.')
flags.DEFINE_integer('inject_epochs', 5,
                     'Number of epochs to inject backdoor.')
flags.DEFINE_integer('student_epochs', 30,
                     'Training epochs of transfer learning.')
flags.DEFINE_integer('student_batch_size', 32,
                     'Batch size of transfer learning.')

flags.DEFINE_integer('nb_tea_classes', 30,
                     'Number of classes in the Teacher task.')
flags.DEFINE_integer('nb_stu_classes', 35,
                     'Number of classes in the Student task.')
flags.DEFINE_integer('trig_opt_batch_size', 32,
                     'Batch size of optimizing trigger.')
flags.DEFINE_integer('trig_opt_steps', 50,
                     'Number of steps to optimize trigger.')
flags.DEFINE_integer('trig_opt_nb_samples', 1000,
                     'Number of samples used to optimize trigger.')
flags.DEFINE_integer('trig_opt_patience', 5,
                     'Patience of trigger optimzation.')
flags.DEFINE_integer('trig_opt_upsample_size', 1,
                     'Upsampling size of trigger optimization.')

# Float flags.
flags.DEFINE_float('pattern_size_ratio', 0.2,
                   ('Percentage of side of a squared trigger compared to'
                    'image size. 0.2 means 4% of image size.'))
flags.DEFINE_float('mse_weight', 0.05, 'Weight to MSE in trigger injection.')
flags.DEFINE_float('teacher_retrain_lr', 0.01,
                   'Learning rate of teacher retraining,')
flags.DEFINE_float('inject_lr', 0.01, 'Learning of backdoor injection.')
flags.DEFINE_float('student_train_lr', 0.01,
                   'Learning rate of transfer learning.')
flags.DEFINE_float('trig_opt_lr', 0.1,
                   'Learning rate of trigger optimization.')
flags.DEFINE_float('trig_opt_init_cost', 1000.0,
                   'Initial cost of trigger optimization.')
flags.DEFINE_float('trig_opt_cost_multiplier', 2.0,
                   'Cost multiplier of trigger optimization.')
flags.DEFINE_float('trig_opt_early_stop_thresh', 1.0,
                   'Early threshold of trigger optimization.')

# Boolean flags.
flags.DEFINE_boolean('trig_opt_save_last', False,
                     'If save last in trigger optimization.')
flags.DEFINE_boolean('trig_opt_early_stop', True,
                     'If using early stop in trigger optimization.')

# String flags.
flags.DEFINE_string('data_path',
                    'data/pubfig.h5',
                    'Path to data file.')
flags.DEFINE_string('model_path',
                    'models/vggface.h5',
                    'Path to original Teacher model file.')
flags.DEFINE_string('intensity_range',
                    'imagenet',
                    'Which preproceesing is used in the model.')


def modify_class(original_model):
    """ Adds one more class to classification layer to include label for
    target class.

    Args:
      original_model: original Teacher model.

    Returns:
      Modified model.
    """
    # Removes the second to last layer to perform transfer learning.
    x = original_model.layers[-3].output
    clf = Dense(FLAGS.nb_tea_classes + 1, name='modified_clf',
                activation='softmax')(x)
    modified_model = Model(inputs=original_model.input, outputs=clf)
    opt = keras.optimizers.Adadelta(lr=FLAGS.teacher_retrain_lr)
    modified_model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

    return modified_model


def build_teacher_training(input_model):
    """ Build model architecture to support dual-loss optimization.

    Args:
      input_model: model input.

    Return:
      teacher: model architecture with dual-loss.
      teacher_botneck: bottleneck model to compute intermediate
        representation used for trigger optimization.
    """
    global graph
    graph = tf.get_default_graph()
    model = keras.models.clone_model(input_model)
    model.set_weights(input_model.get_weights())

    opt = keras.optimizers.Adadelta(lr=FLAGS.inject_lr)
    norm_input = Input(model.input_shape[1:], name='adv_input')
    x = norm_input
    for layer in model.layers[1:]:
        x = layer(x)
    # Dual-loss model to inject trigger.
    teacher = Model(inputs=[norm_input, model.input],
                    outputs=[x, model.layers[-5].get_output_at(0)])
    teacher.compile(loss=['categorical_crossentropy', 'mse'],
                    loss_weights=[1, FLAGS.mse_weight],
                    metrics=['accuracy', 'mse'],
                    optimizer=opt)
    # Single input bottleneck model for trigger optimization.
    teacher_botneck = Model(inputs=model.input,
                            outputs=model.layers[-5].get_output_at(0))

    return teacher, teacher_botneck


def generate_pattern(pattern_gen, gen):
    """Generates pattern from an instance of Patterngenerator class and
       data generator.

    Args:
      pattern_gen: an instance of Patterngenerator.
      gen: data generator.

    Returns:
      Optimized pattern.
    """

    # Initializes with random pattern.
    input_shape = (FLAGS.img_width, FLAGS.img_height, FLAGS.nb_channels)
    pattern = np.random.random(input_shape) * 255.0

    # Optimizes trigger pattern.
    pattern, _, _, _ = pattern_gen.generate(
        gen=gen, pattern_init=pattern)

    return pattern


def build_trig_opt_data_generator(norm_x, tgt_x, bottleneck_model):
    """Builds a data generator for trigger optimization.

    Args:
      norm_x: normal samples.
      tgt_x: target data.
      bottleneck_model: Bottleneck model to compute intermediate values.

    Returns:
      A data generator with normal and target samples.
    """
    tgt_val = bottleneck_model.predict(tgt_x)

    while True:
        norm_idx = np.random.choice(
            norm_x.shape[0], FLAGS.trig_opt_batch_size)
        tgt_idx = np.random.choice(
            tgt_val.shape[0], FLAGS.trig_opt_batch_size)
        norm_data = norm_x[norm_idx]
        tgt_data = tgt_val[tgt_idx]
        yield norm_data, tgt_data


def trigger_opt(x_train_tea, target_data, bottleneck_model, mask):
    """Trigger optimization function.

    Args:
      x_train_tea: Teacher training dataset.
      target_data: Target data.
      bottleneck_model: Bottleneck model to compute intermediate representation.
      mask: Mask of trigger.

    Returns:
      Pattern of optimized trigger.
    """
    trig_opt_data_gen = build_trig_opt_data_generator(
        x_train_tea, target_data, bottleneck_model)

    pattern_gen = PatternGenerator(
        bottleneck_model, mask, intensity_range=FLAGS.intensity_range,
        input_shape=(FLAGS.img_width, FLAGS.img_height, FLAGS.nb_channels),
        init_cost=FLAGS.trig_opt_init_cost, steps=FLAGS.trig_opt_steps,
        lr=FLAGS.trig_opt_lr,
        upsample_size=FLAGS.trig_opt_upsample_size,
        mini_batch=FLAGS.trig_opt_nb_samples //
        FLAGS.trig_opt_batch_size,
        patience=FLAGS.trig_opt_patience,
        cost_multiplier=FLAGS.trig_opt_cost_multiplier,
        img_color=FLAGS.nb_channels, batch_size=FLAGS.trig_opt_batch_size,
        verbose=2,
        save_last=FLAGS.trig_opt_save_last,
        early_stop=FLAGS.trig_opt_early_stop,
        early_stop_threshold=FLAGS.trig_opt_early_stop_thresh,
        early_stop_patience=5 * FLAGS.trig_opt_patience)

    pattern = generate_pattern(pattern_gen, trig_opt_data_gen)

    return pattern


def inject_trigger(raw_x, mask, pattern):
    """ Inject trigger to input data.

    Args:
      raw_x: Raw images without preproccessing.
      mask: Trigger's mask.
      pattern: Trigger's pattern.

    Returns:
      Poisoned samples.
    """
    x_adv = []
    for i in range(raw_x.shape[0]):
        adv_img = np.copy(raw_x[i])
        bool_mask = mask != 0
        adv_img[bool_mask] = 0
        adv_img[bool_mask] += pattern[bool_mask]
        x_adv.append(adv_img)
    x_adv = np.stack(x_adv)
    return x_adv


def generate_poisoned_data(x_train_tea, y_train_tea, mask, pattern,
                           bottleneck_model, target_y_tea):
    """Generates poisoned dataset.

    Args:
      x_train_tea: Teacher training data.
      y_train_tea: Label of Teacher training data.
      mask: Mask of trigger.
      pattern: Optimized pattern of trigger.
      bottleneck_model: Bottleneck model.
      target_y_tea: Label of target class in Teacher task.

    Returns:
      x_adv: Adversarial samples by adding trigger to training samples.
      y_adv: Intermediate representation of the target data, i.e., what
        adversarial samples should be mimicking in trigger injection.
    """
    x_tgt = x_train_tea[
        np.argmax(y_train_tea, axis=1) == target_y_tea]
    # Reverses training data to the state before preprocessing.
    x_train_teacher_raw = latent_utils.imagenet_reverse_preprocessing(
        np.copy(x_train_tea), data_format='channels_last')
    # Adds trigger in raw image space.
    x_adv_raw = inject_trigger(x_train_teacher_raw, mask, pattern)
    # Preproccesses altered images.
    x_adv = vgg19_preprocess(np.copy(x_adv_raw))

    # Computes intermediate representation of target data.
    y_adv = bottleneck_model.predict(x_tgt)
    y_adv = y_adv[np.random.choice(y_adv.shape[0], x_adv.shape[0])]

    return x_adv, y_adv


def poison_data_generator(x_train_tea, x_adv, y_train_tea, y_adv):
    """ Combined normal samples and poisoned samples.

    Args:
      x_train_tea: Teacher clean training samples.
      x_adv: Adversarial samples.
      y_train_tea: Label of clean samples.
      y_adv: Intermediate representation of target data that
       adversarial samples should be mapped to through backdoor injection.

    Returns:
      A data generator.
    """

    while True:
        rand_idx = np.random.choice(x_train_tea.shape[0],
                                    FLAGS.inject_batch_size)
        x_norm_batch = x_train_tea[rand_idx]
        y_norm_batch = y_train_tea[rand_idx]
        x_adv_batch = x_adv[rand_idx]
        y_adv_batch = y_adv[rand_idx]
        yield [x_norm_batch, x_adv_batch], [y_norm_batch, y_adv_batch]


def get_student_model(teacher):
    """Transfer weights from Teacher model to build a Student model.

    Args:
      teacher: Teacher model.

    Returns:
      Student model.
    """
    # Load the Teacher model's original architecture only, weights will be
    # replaced.
    student = keras.models.load_model(FLAGS.model_path)

    # Transfers weights from the Teacher model.
    for stu_layer in student.layers:
        stu_weights = stu_layer.get_weights()
        if stu_weights:
            for tea_layer in teacher.layers[:-4]:
                tea_weights = tea_layer.get_weights()
                if tea_weights:
                    if stu_layer.name == tea_layer.name:
                        stu_layer.set_weights(tea_weights)

    # Appends layers.
    x = student.layers[-5].output
    x = Dense(4096, name='added_dense', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(FLAGS.nb_stu_classes,
              name='student_clf',
              activation='softmax')(x)
    student = Model(inputs=student.input, outputs=x)

    # Freezes 4096 dense + last 1000 clf.
    for layer in student.layers[:-3]:
        layer.trainable = False

    opt = keras.optimizers.Adadelta(lr=FLAGS.student_train_lr)
    student.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    return student


def evaluate_attack(x, mask, pattern, student, target_y):
    """Evaluates attack success rate.

    Args:
      x: input samples.
      mask: trigger's mask.
      pattern: trigger's pattern.
      student: Student model.
      target_y: Targeted class in Student task.

    Returns:
      Attack success rate, the percentage of poisoned samples misclassified
        into the target label.
    """
    # Revert x to raw input space.
    raw_x = latent_utils.imagenet_reverse_preprocessing(
        np.copy(x), data_format='channels_last')
    # Add trigger to raw images.
    x_adv = inject_trigger(raw_x, mask, pattern)
    # Preproccessed infected data.
    x_adv = vgg19_preprocess(np.copy(x_adv))
    # Makes inference and compute metrics.
    attack_success = np.mean(
        np.argmax(student.predict(x_adv), axis=1) == target_y)

    return attack_success


def main():
    # GPU settings.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    latent_utils.fix_gpu_memory()

    # Loads data. Note they're already preproccssed.
    with h5py.File(FLAGS.data_path, 'r') as hf:
        x_train_tea = hf['x_train_tea'][:]
        y_train_tea = hf['y_train_tea'][:]
        x_test_tea = hf['x_test_tea'][:]
        y_test_tea = hf['y_test_tea'][:]
        x_train_stu = hf['x_train_stu'][:]
        y_train_stu = hf['y_train_stu'][:]
        x_test_stu = hf['x_test_stu'][:]
        y_test_stu = hf['y_test_stu'][:]
        target_data = hf['target_data'][:]

    # Loads original Teacher model.
    teacher = keras.models.load_model(FLAGS.model_path)

    # Modifies the classification layer to include target class.
    teacher = modify_class(teacher)

    # Retrains the Teacher model.
    teacher.fit(x_train_tea,
                y_train_tea,
                epochs=FLAGS.teacher_retrain_epochs,
                verbose=2)
    _, retrain_acc = teacher.evaluate(x_test_tea, y_test_tea)
    print('Retrained Teacher model accuracy: %f' % retrain_acc)

    teacher, teacher_botneck = build_teacher_training(teacher)

    # Optimizes trigger.
    print('Optimizing trigger pattern')
    mask, _ = latent_utils.construct_mask(
        num_patterns=1, image_dim=FLAGS.img_width,
        channel_num=FLAGS.nb_channels,
        pattern_size=int(FLAGS.img_width *
                         FLAGS.pattern_size_ratio),
        randomize=False)
    pattern = trigger_opt(x_train_tea, target_data, teacher_botneck, mask)

    # Generates adversarial samples.
    x_adv, y_adv = generate_poisoned_data(
        x_train_tea, y_train_tea, mask, pattern,
        teacher_botneck, FLAGS.target_y_tea)

    # Combines adversarial examples with clean samples.
    posioned_data_gen = poison_data_generator(
        x_train_tea, x_adv, y_train_tea, y_adv)

    # Injects the latent backdoor.
    print('Injecting latent backdoor')
    for e in range(FLAGS.inject_epochs):
        print('Injection epoch %d' % e)
        teacher.fit_generator(
            posioned_data_gen,
            steps_per_epoch=x_train_tea.shape[0] //
            FLAGS.inject_batch_size,
            epochs=1,
            verbose=0)

    # Student side: Performs Transfer learning.
    student = get_student_model(teacher)

    # Student side: trains student model on clean student training data.
    print('Training Student')
    student.fit(x_train_stu, y_train_stu,
                batch_size=FLAGS.student_batch_size,
                epochs=FLAGS.student_epochs,
                verbose=2)
    _, student_acc = student.evaluate(x_test_stu, y_test_stu,
                                      verbose=0)
    print('Student model accuracy: %f' % student_acc)

    # Evaluates attack.
    attack_success = evaluate_attack(x_test_tea, mask, pattern,
                                     student, FLAGS.target_y_stu)
    print('Attack success rate on the Student model: %f' % attack_success)


if __name__ == '__main__':
    main()
