﻿# _*_ coding:utf-8 _*_
import tensorflow as tf
import os
import logging
import numpy as np
import SimpleITK
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_string('load_model', "20190715-1643",
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_string('code_tensor_name', "GPU_0/random_normal_1:0", "code tensor name")
tf.flags.DEFINE_string('f_tensor_name', "GPU_0/Reshape_4:0", "f tensor name")
tf.flags.DEFINE_string('m_tensor_name', "GPU_0/Reshape_5:0", "m tensor name")
tf.flags.DEFINE_string('j_f_tensor_name', "GPU_3/D_F_1/conv5/conv5/BiasAdd:0", "j_f tensor name")
tf.flags.DEFINE_integer('epoch_steps', 160000, '')
tf.flags.DEFINE_integer('epochs', 1, '')
tf.flags.DEFINE_float('min_j_f', 0.85, '')
tf.flags.DEFINE_float('max_count', 50, '')
tf.flags.DEFINE_float('mae', 0.048, '')


def get_mask_from_f(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    gray = cv2.GaussianBlur(img, (3, 3), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    c_list = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = c_list[-2], c_list[-1]
    cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=-1)
    return np.asarray(1.0 - img / 255.0, dtype="float32")


def train():
    if FLAGS.load_model is not None:
        if FLAGS.savefile is not None:
            checkpoints_dir = FLAGS.savefile + "/checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        else:
            checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")

    else:
        print("<load_model> is None.")
        return
    try:
        os.makedirs("./mydata/F_and_M/F_jpg")
        os.makedirs("./mydata/F_and_M/Temp/F")
        os.makedirs("./mydata/F_and_M/Temp/M1")
        os.makedirs("./mydata/F_and_M/Temp/M2")
        os.makedirs("./mydata/F_and_M/F")
        os.makedirs("./mydata/F_and_M/M")
    except os.error:
        pass
    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
    model_checkpoint_path = checkpoint.model_checkpoint_path
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    meta_graph_path = model_checkpoint_path + ".meta"
    saver = tf.train.import_meta_graph(meta_graph_path)

    graph = tf.get_default_graph()
    code_rm = tf.get_default_graph().get_tensor_by_name(FLAGS.code_tensor_name)
    f_rm = tf.get_default_graph().get_tensor_by_name(FLAGS.f_tensor_name)
    mask_rm = tf.get_default_graph().get_tensor_by_name(FLAGS.m_tensor_name)
    j_f_rm = tf.get_default_graph().get_tensor_by_name(FLAGS.j_f_tensor_name)

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, latest_checkpoint)
        index = 0
        while index <= FLAGS.epoch_steps * FLAGS.epochs:

            count = 0
            best_j_f = -1000.0
            while True:
                code = sess.run(code_rm)
                f, m, j_f = sess.run([f_rm, mask_rm, j_f_rm], feed_dict={code_rm: code})
                j_f = np.mean(np.asarray(j_f))

                if j_f >= FLAGS.min_j_f: break

                jpg_f = np.concatenate([np.asarray(f)[0, :, :, 0:1] * 255, np.asarray(f)[0, :, :, 0:1] * 255,
                                        np.asarray(f)[0, :, :, 0:1] * 255], axis=-1)
                cv2.imwrite("./mydata/F_and_M/Temp/F/f_" + str(
                    index) + "_" + str(count) + ".jpg", jpg_f)

                m_arr_1 = get_mask_from_f(
                    "./mydata/F_and_M/Temp/F/f_" + str(
                        index) + "_" + str(count) + ".jpg")
                m_arr_2 = np.asarray(m)[0, :, :, 0].astype('float32')

                mae = np.mean(np.abs(m_arr_1 - m_arr_2))

                if mae <= FLAGS.mae: break

                if j_f > best_j_f:
                    best_j_f = j_f
                    best_f = f
                    best_m = m

                if count >= FLAGS.max_count:
                    f = best_f
                    m = best_m
                    break

                count = count + 1

            jpg_f = np.concatenate([np.asarray(f)[0, :, :, 0:1] * 255, np.asarray(f)[0, :, :, 0:1] * 255,
                                    np.asarray(f)[0, :, :, 0:1] * 255], axis=-1)
            cv2.imwrite("./mydata/F_and_M/F_jpg/" + str(index) + ".jpg",
                        jpg_f)
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(f)[0, :, :, 0]),
                                 "./mydata/F_and_M/F/" + str(
                                     index) + ".tiff")
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(m)[0, :, :, 0]),
                                 "./mydata/F_and_M/M/" + str(
                                     index) + ".tiff")
            index += 1


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
