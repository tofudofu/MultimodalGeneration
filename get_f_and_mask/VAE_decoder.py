# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class GDecoder:
    def __init__(self, name, ngf=64, keep_prob=1.0, output_channl=1):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.keep_prob = keep_prob
        self.output_channl = output_channl

    def __call__(self, DC_input):
        """
        Args:
          input: batch_size x width x height x N
        Returns:
          output: same size as input
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope("dense0", reuse=self.reuse):
                dense0 = tf.layers.dense(DC_input, units=DC_input.get_shape().as_list()[0] * 6 * 5 * self.ngf,
                                         name="dense0")
            with tf.variable_scope("dense1", reuse=self.reuse):
                dense1 = tf.layers.dense(dense0, units=DC_input.get_shape().as_list()[0] * 6 * 5 * 12 * self.ngf,
                                         name="dense0")
                dense1 = tf.reshape(dense1, shape=[DC_input.get_shape().as_list()[0], 6, 5, 12 * self.ngf])
            # 6,5
            with tf.variable_scope("conv0_1", reuse=self.reuse):
                conv0_1 = tf.layers.conv2d(inputs=dense1, filters=12 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=1.0 / (9.0 * 12 * self.ngf), stddev=0.000001,
                                               dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_1')
                norm0_1 = ops.norm(conv0_1)
                relu0_1 = ops.relu(norm0_1)
            # 6,5
            with tf.variable_scope("deconv0_1_r", reuse=self.reuse):
                resize0_1 = ops.uk_resize(relu0_1, reuse=self.reuse, output_size=[12, 9], name='resize')
                deconv0_1_r = tf.layers.conv2d(inputs=resize0_1, filters=8 * self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=1.0 / (9.0 * 12 * self.ngf), stddev=0.000001,
                                                   dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               name='deconv0_1_r')
                deconv0_1_norm1_r = ops.norm(deconv0_1_r)
                deconv0_1_relu1 = ops.relu(deconv0_1_norm1_r)
            # 12,9
            with tf.variable_scope("conv0_2", reuse=self.reuse):
                conv0_2 = tf.layers.conv2d(inputs=deconv0_1_relu1, filters=8 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=1.0 / (9.0 * 8 * self.ngf), stddev=0.000001,
                                               dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_2')
                norm0_2 = ops.norm(conv0_2)
                relu0_2 = ops.relu(norm0_2)
            # 12,9
            with tf.variable_scope("deconv0_2_r", reuse=self.reuse):
                resize0_2 = ops.uk_resize(relu0_2, reuse=self.reuse, output_size=[23, 18], name='resize')
                deconv0_2_r = tf.layers.conv2d(inputs=resize0_2, filters=6 * self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=1.0 / (9.0 * 8 * self.ngf), stddev=0.000001,
                                                   dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               name='deconv0_2_r')
                deconv0_2_norm1_r = ops.norm(deconv0_2_r)
                deconv0_2_relu1 = ops.relu(deconv0_2_norm1_r)
            # 23, 18
            with tf.variable_scope("conv0_3", reuse=self.reuse):
                conv0_3 = tf.layers.conv2d(inputs=deconv0_2_relu1, filters=6 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                               dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_3')
                norm0_3 = ops.norm(conv0_3)
                relu0_3 = ops.relu(norm0_3)
            # 23, 18
            with tf.variable_scope("deconv0_3_r", reuse=self.reuse):
                resize0_3 = ops.uk_resize(relu0_3, reuse=self.reuse, name='resize')
                deconv0_3_r = tf.layers.conv2d(inputs=resize0_3, filters=6 * self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                                   dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               name='deconv0_3_r')
                deconv0_3_norm1_r = ops.norm(deconv0_3_r)
                add0 = ops.relu(deconv0_3_norm1_r)
            # 46, 36
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=add0, filters=6 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                             dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops.norm(conv1)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("deconv1_r", reuse=self.reuse):
                resize1 = ops.uk_resize(relu1, reuse=self.reuse, name='resize')
                deconv1_r = tf.layers.conv2d(inputs=resize1, filters=4 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv1_r')
                deconv1_norm1_r = ops.norm(deconv1_r)
                add1 = ops.relu(deconv1_norm1_r)
            with tf.variable_scope("add1_conv1", reuse=self.reuse):
                add1_conv1 = tf.layers.conv2d(inputs=add1, filters=4 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add1_conv1')
                add1_norm1 = ops.norm(add1_conv1)
                add1_relu1 = ops.relu(add1_norm1)
            with tf.variable_scope("add1_conv2", reuse=self.reuse):
                add1_conv2 = tf.layers.conv2d(inputs=add1_relu1, filters=4 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add1_conv2')
                add1_norm2 = ops.norm(add1_conv2)
                add1_relu2 = ops.relu(add1_norm2)
            with tf.variable_scope("deconv2_r", reuse=self.reuse):
                resize2 = ops.uk_resize(add1_relu2, reuse=self.reuse, name='resize')
                deconv2_r = tf.layers.conv2d(inputs=resize2, filters=2 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv2_r')
                deconv2_norm1_r = ops.norm(deconv2_r)
                add2 = ops.relu(deconv2_norm1_r)
            with tf.variable_scope("add2_conv1", reuse=self.reuse):
                add2_conv1 = tf.layers.conv2d(inputs=add2, filters=2 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add2_conv1')
                add2_norm1 = ops.norm(add2_conv1)
                add2_relu1 = ops.relu(add2_norm1)
            with tf.variable_scope("add2_conv2", reuse=self.reuse):
                add2_conv = tf.layers.conv2d(inputs=add2_relu1, filters=2 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='add2_conv2')
                add2_norm2 = ops.norm(add2_conv)
                add2_relu2 = ops.relu(add2_norm2)
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=add2_relu2, filters=self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops.norm(conv2)
                relu2 = ops.relu(norm2)
            with tf.variable_scope("lastconv", reuse=self.reuse):
                lastconv = tf.layers.conv2d(inputs=relu2, filters=self.output_channl, kernel_size=3, strides=1,
                                            padding="SAME",
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(
                                                mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                            bias_initializer=tf.constant_initializer(0.0), name='lastconv')
                lastnorm = ops.norm(lastconv)
                output = tf.nn.sigmoid(lastnorm)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
