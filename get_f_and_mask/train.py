﻿# _*_ coding:utf-8 _*_
import tensorflow as tf
from VAE_model import VAE
from datetime import datetime
import os
import logging
import numpy as np
import SimpleITK

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('savefile', None, 'Checkpoint save dir')
tf.flags.DEFINE_integer('log_level', 10, 'CRITICAL = 50,ERROR = 40,WARNING = 30,INFO = 20,DEBUG = 10,NOTSET = 0')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size, default: 1')
tf.flags.DEFINE_list('image_size', [184, 144, 1], 'image size,')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate for Adam')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer')
tf.flags.DEFINE_string('X', '../mydata/BRATS2015/trainT1', 'files path')
tf.flags.DEFINE_string('Y', '../mydata/BRATS2015/trainT2', 'files path')
tf.flags.DEFINE_string('Z', '../mydata/BRATS2015/trainT1c', 'files path')
tf.flags.DEFINE_string('W', '../mydata/BRATS2015/trainFlair', 'files path')
tf.flags.DEFINE_string('L', '../mydata/BRATS2015/trainLabel', 'files path')
tf.flags.DEFINE_string('X_test', '../mydata/BRATS2015/testT1', 'files path')
tf.flags.DEFINE_string('Y_test', '../mydata/BRATS2015/testT2', 'files path')
tf.flags.DEFINE_string('Z_test', '../mydata/BRATS2015/testT1c', 'files path')
tf.flags.DEFINE_string('W_test', '../mydata/BRATS2015/testFlair', 'files path')
tf.flags.DEFINE_string('L_test', '../mydata/BRATS2015/testLabel', 'files path')
tf.flags.DEFINE_string('load_model', "20190715-1643",
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint', None, "default: None")
tf.flags.DEFINE_bool('step_clear', False,
                     'if continue training, step clear, default: True')
tf.flags.DEFINE_integer('epoch', 100, 'default: 100')


def read_file(l_path, Label_train_files, index):
    train_range = len(Label_train_files)
    L_img = SimpleITK.ReadImage(l_path + "/" + Label_train_files[index % train_range])
    L_arr_ = SimpleITK.GetArrayFromImage(L_img)
    L_arr_ = L_arr_.astype('float32')
    return L_arr_


def read_filename(path, shuffle=True):
    files = os.listdir(path)
    files_ = np.asarray(files)
    if shuffle == True:
        index_arr = np.arange(len(files_))
        np.random.shuffle(index_arr)
        files_ = files_[index_arr]
    return files_


def average_gradients(grads_list):
    average_grads = []
    for grad_and_vars in zip(*grads_list):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.device("/cpu:0"):
        if FLAGS.load_model is not None:
            if FLAGS.savefile is not None:
                checkpoints_dir = FLAGS.savefile + "/checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
            else:
                checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        else:
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            if FLAGS.savefile is not None:
                checkpoints_dir = FLAGS.savefile + "/checkpoints/{}".format(current_time)
            else:
                checkpoints_dir = "checkpoints/{}".format(current_time)
            try:
                os.makedirs(checkpoints_dir + "/samples")
            except os.error:
                pass

        for attr, value in FLAGS.flag_values_dict().items():
            logging.info("%s\t:\t%s" % (attr, str(value)))

        graph = tf.Graph()
        with graph.as_default():
            vae = VAE(FLAGS.image_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.ngf)
            input_shape = [int(FLAGS.batch_size / 4), FLAGS.image_size[0], FLAGS.image_size[1], FLAGS.image_size[2]]
            FG_optimizer, MG_optimizer, D_optimizer = vae.optimize()

            FG_grad_list = []
            MG_grad_list = []
            D_grad_list = []
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device("/gpu:0"):
                    with tf.name_scope("GPU_0"):
                        m_0 = tf.placeholder(tf.float32, shape=input_shape)
                        l_m_0 = tf.placeholder(tf.float32, shape=input_shape)
                        image_list_0, code_list_0, j_list_0, loss_list_0 = vae.model(l_m_0, m_0)
                        tensor_name_dirct_0 = vae.tenaor_name
                        variables_list_0 = vae.get_variables()
                        FG_grad_0 = FG_optimizer.compute_gradients(loss_list_0[0], var_list=variables_list_0[0])
                        MG_grad_0 = MG_optimizer.compute_gradients(loss_list_0[1], var_list=variables_list_0[1])
                        D_grad_0 = D_optimizer.compute_gradients(loss_list_0[2], var_list=variables_list_0[2])
                        FG_grad_list.append(FG_grad_0)
                        MG_grad_list.append(MG_grad_0)
                        D_grad_list.append(D_grad_0)
                with tf.device("/gpu:1"):
                    with tf.name_scope("GPU_1"):
                        m_1 = tf.placeholder(tf.float32, shape=input_shape)
                        l_m_1 = tf.placeholder(tf.float32, shape=input_shape)
                        image_list_1, code_list_1, j_list_1, loss_list_1 = vae.model(l_m_1, m_1)
                        tensor_name_dirct_1 = vae.tenaor_name
                        variables_list_1 = vae.get_variables()
                        FG_grad_1 = FG_optimizer.compute_gradients(loss_list_1[0], var_list=variables_list_1[0])
                        MG_grad_1 = MG_optimizer.compute_gradients(loss_list_1[1], var_list=variables_list_1[1])
                        D_grad_1 = D_optimizer.compute_gradients(loss_list_1[2], var_list=variables_list_1[2])
                        FG_grad_list.append(FG_grad_1)
                        MG_grad_list.append(MG_grad_1)
                        D_grad_list.append(D_grad_1)
                with tf.device("/gpu:2"):
                    with tf.name_scope("GPU_2"):
                        m_2 = tf.placeholder(tf.float32, shape=input_shape)
                        l_m_2 = tf.placeholder(tf.float32, shape=input_shape)
                        image_list_2, code_list_2, j_list_2, loss_list_2 = vae.model(l_m_2, m_2)
                        tensor_name_dirct_2 = vae.tenaor_name
                        variables_list_2 = vae.get_variables()
                        FG_grad_2 = FG_optimizer.compute_gradients(loss_list_2[0], var_list=variables_list_2[0])
                        MG_grad_2 = MG_optimizer.compute_gradients(loss_list_2[1], var_list=variables_list_2[1])
                        D_grad_2 = D_optimizer.compute_gradients(loss_list_2[2], var_list=variables_list_2[2])
                        FG_grad_list.append(FG_grad_2)
                        MG_grad_list.append(MG_grad_2)
                        D_grad_list.append(D_grad_2)
                with tf.device("/gpu:3"):
                    with tf.name_scope("GPU_3"):
                        m_3 = tf.placeholder(tf.float32, shape=input_shape)
                        l_m_3 = tf.placeholder(tf.float32, shape=input_shape)
                        image_list_3, code_list_3, j_list_3, loss_list_3 = vae.model(l_m_3, m_3)
                        tensor_name_dirct_3 = vae.tenaor_name
                        variables_list_3 = vae.get_variables()
                        FG_grad_3 = FG_optimizer.compute_gradients(loss_list_3[0], var_list=variables_list_3[0])
                        MG_grad_3 = MG_optimizer.compute_gradients(loss_list_3[1], var_list=variables_list_3[1])
                        D_grad_3 = D_optimizer.compute_gradients(loss_list_3[2], var_list=variables_list_3[2])
                        FG_grad_list.append(FG_grad_3)
                        MG_grad_list.append(MG_grad_3)
                        D_grad_list.append(D_grad_3)

            FG_ave_grad = average_gradients(FG_grad_list)
            MG_ave_grad = average_gradients(MG_grad_list)
            D_ave_grad = average_gradients(D_grad_list)
            FG_optimizer_op = FG_optimizer.apply_gradients(FG_ave_grad)
            MG_optimizer_op = MG_optimizer.apply_gradients(MG_ave_grad)
            D_optimizer_op = D_optimizer.apply_gradients(D_ave_grad)
            optimizers = [FG_optimizer_op, MG_optimizer_op, D_optimizer_op]

            saver = tf.train.Saver()

        with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if FLAGS.load_model is not None:
                logging.info("restore model:" + FLAGS.load_model)
                if FLAGS.checkpoint is not None:
                    model_checkpoint_path = checkpoints_dir + "/model.ckpt-" + FLAGS.checkpoint
                    latest_checkpoint = model_checkpoint_path
                else:
                    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                    model_checkpoint_path = checkpoint.model_checkpoint_path
                    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
                logging.info("model checkpoint path:" + model_checkpoint_path)
                meta_graph_path = model_checkpoint_path + ".meta"
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, latest_checkpoint)
                if FLAGS.step_clear == True:
                    step = 0
                else:
                    step = int(meta_graph_path.split("-")[2].split(".")[0])
            else:
                sess.run(tf.global_variables_initializer())
                step = 0
            sess.graph.finalize()
            logging.info("start step:" + str(step))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                logging.info("tensor_name_dirct:\n" + str(tensor_name_dirct_0))
                m_train_files = read_filename(FLAGS.L)
                index = 0
                epoch = 0
                while not coord.should_stop() and epoch <= FLAGS.epoch:

                    train_true_m = []
                    train_true_l_m = []
                    for b in range(FLAGS.batch_size):
                        train_m_arr = read_file(np.asarray([FLAGS.X, FLAGS.Y, FLAGS.Z, FLAGS.W])[np.random.randint(4)],
                                                m_train_files, index).reshape(FLAGS.image_size)
                        train_l_m_arr = read_file(FLAGS.L, m_train_files, index).reshape(FLAGS.image_size)
                        train_true_m.append(train_m_arr)
                        train_true_l_m.append(train_l_m_arr)
                        epoch = int(index / len(m_train_files))
                        index = index + 1

                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": start-------------")
                    sess.run(optimizers,
                             feed_dict={
                                 m_0: np.asarray(train_true_m)[0:1, :, :, :],
                                 m_1: np.asarray(train_true_m)[1:2, :, :, :],
                                 m_2: np.asarray(train_true_m)[2:3, :, :, :],
                                 m_3: np.asarray(train_true_m)[3:4, :, :, :],

                                 l_m_0: np.asarray(train_true_l_m)[0:1, :, :, :],
                                 l_m_1: np.asarray(train_true_l_m)[1:2, :, :, :],
                                 l_m_2: np.asarray(train_true_l_m)[2:3, :, :, :],
                                 l_m_3: np.asarray(train_true_l_m)[3:4, :, :, :],

                             })
                    logging.info(
                        "-----------train epoch " + str(epoch) + ", step " + str(step) + ": end-------------")

                    step += 1
            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logging.basicConfig(level=FLAGS.log_level)
    tf.app.run()
