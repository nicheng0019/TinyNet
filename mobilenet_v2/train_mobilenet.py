import os
import gzip
import numpy as np
import tensorflow as tf
from mobilenet_v2 import *
from convnet_builder import *
import arg_parsing


_IMAGE_SIZE = 28
_NUM_CHANNELS = 1

def _extract_images(filename, num_images):
  """Extract the images into a numpy array.

  Args:
    filename: The path to an MNIST images file.
    num_images: The number of images in the file.

  Returns:
    A numpy array of shape [number_of_images, height, width, channels].
  """
  print('Extracting images from: ', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(
        _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  return data


def _extract_labels(filename, num_labels):
  """Extract the labels into a vector of int64 label IDs.

  Args:
    filename: The path to an MNIST labels file.
    num_labels: The number of labels in the file.

  Returns:
    A numpy array of shape [number_of_labels]
  """
  print('Extracting labels from: ', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

def read_and_decode(data_filename, labels_filename, num_images):
    images = _extract_images(data_filename, num_images)
    labels = _extract_labels(labels_filename, num_images)

    return images, labels

def generate_batch(image_batch, label_patch, batch_size, max_step, shuffle=True):
    """Generate batch from X_train/X_test and y_train/y_test using a python DataGenerator"""
    n = 0
    new_epoch = True
    start_idx = 0
    mask = None
    train_data_len = image_batch.shape[0]
    while n < max_step:
        if new_epoch:
            start_idx = 0
            if shuffle:
                mask = np.random.choice(train_data_len, train_data_len, replace=False)
            else:
                mask = np.arange(train_data_len)
            new_epoch = False

        # Batch mask selection
        X_batch = image_batch[mask[start_idx:start_idx + batch_size]]
        y_batch = label_patch[mask[start_idx:start_idx + batch_size]]
        start_idx += batch_size

        if start_idx >= train_data_len:
            new_epoch = True
            mask = None

        n += 1
        yield X_batch, y_batch

def main(args=None):
    args = arg_parsing.ArgParser().parse_args(args)

    num_classes = args.num_classes
    min_depth = args.min_depth
    depth_multiplier = args.depth_multiplier
    conv_defs = None
    spatial_squeeze = True
    reuse = None
    global_pool = False
    learning_rate = args.learning_rate
    model_dir = args.model_dir
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    train_data_filename = args.train_data_filename
    train_label_filename = args.train_label_filename
    test_data_filename = args.test_data_filename
    test_label_filename = args.test_label_filename
    train_sample_num = args.train_sample_num
    test_sample_num = args.test_sample_num
    keep_last_n_checkpoints = args.keep_last_n_checkpoints

    train_images, train_labels = read_and_decode(train_data_filename, train_label_filename, train_sample_num)
    test_images, test_labels = read_and_decode(test_data_filename, test_label_filename, test_sample_num)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        inputs = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS], name="input")
        labels = tf.placeholder(tf.int32, shape=[None], name="label")
        is_trainging = tf.placeholder(tf.bool, name="is_training")
        dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")

        cnn = ConvNetBuilder(input_op=inputs, input_nchan=1, phase_train=True, use_tf_layers=False, data_format="NHWC")

        logits, end_points = mobilenet_v2(cnn, num_classes=num_classes, dropout_keep_prob=dropout_keep_prob, is_training=is_trainging,
            min_depth=min_depth, depth_multiplier=depth_multiplier, conv_defs=conv_defs, prediction_fn=tf.contrib.layers.softmax,
            spatial_squeeze=spatial_squeeze, reuse=reuse, scope='MobilenetV2', global_pool=global_pool)

        predictions = tf.argmax(logits, 1, name='predictions')
        predictions = tf.cast(predictions, dtype=tf.float32)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        out_softmax = tf.nn.softmax(logits)
        out_argmax = tf.argmax(out_softmax, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, out_argmax), tf.float32))

        saver = tf.train.Saver(max_to_keep=keep_last_n_checkpoints)
        save_path = os.path.join(model_dir, 'model.ckpt')

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            last_checkpoint = tf.train.latest_checkpoint(model_dir)
            if last_checkpoint:
                saver.restore(sess, last_checkpoint)

            for epoch in range(max_epoch):
                print("epoch: ", epoch)
                max_step = train_sample_num // batch_size
                if 0 != (train_sample_num % batch_size):
                    max_step += 1

                acc_list = []
                loss_list = []
                for image_batch, label_batch in generate_batch(train_images, train_labels, batch_size, max_step, shuffle=True):
                    _, loss_, accuracy_ = sess.run([optimizer, loss, accuracy], feed_dict={inputs: image_batch, labels: label_batch, is_trainging: True, dropout_keep_prob: 0.5})

                    loss_list.append(loss_)
                    acc_list.append(accuracy_)

                loss_list = np.array(loss_list)
                acc_list = np.array(acc_list)
                if True:#epoch % 1 == 0:
                    print("train step {:}, loss is {:.4}, accuracy is {:.4}".format(epoch, np.mean(loss_list), np.mean(acc_list)))

                    max_step = test_sample_num // batch_size
                    if 0 != (test_sample_num % batch_size):
                        max_step += 1

                    acc_list = []
                    loss_list = []
                    for image_batch, label_batch in generate_batch(test_images, test_labels, batch_size, max_step,
                                                                   shuffle=False):
                        loss_, accuracy_ = sess.run([loss, accuracy],
                                                       feed_dict={inputs: image_batch, labels: label_batch,
                                                                  is_trainging: False, dropout_keep_prob: 1.0})

                        loss_list.append(loss_)
                        acc_list.append(accuracy_)

                    loss_list = np.array(loss_list)
                    acc_list = np.array(acc_list)
                    print("test loss is {:.4}, accuracy is {:.4}".format(np.mean(loss_list), np.mean(acc_list)))

                    saver.save(sess, save_path, global_step=epoch)


if __name__ == '__main__':
    main()
