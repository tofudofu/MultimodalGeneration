# _*_ coding:utf-8 _*_
import tensorflow as tf
from VAE_discriminator import Discriminator
from VAE_feature_discriminator import FeatureDiscriminator
from VAE_encoder import VEncoder
from VAE_decoder import VDecoder
from encoder import Encoder
from decoder import Decoder


class GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
                 ngf=64,
                 ):
        """
        Args:
          input_size：list [N, H, W, C]
          batch_size: integer, batch size
          learning_rate: float, initial learning rate for Adam
          ngf: number of base gen filters in conv layer
        """
        self.learning_rate = learning_rate
        self.input_shape = [int(batch_size / 4), image_size[0], image_size[1], image_size[2]]
        self.ones = tf.ones(self.input_shape, name="ones")
        self.tenaor_name = {}

        self.EC_MASK = Encoder('EC_MASK', ngf=ngf)
        self.DC_MASK = Decoder('DC_MASK', ngf=ngf, output_channl=2)

        self.EC_F = VEncoder('EC_F', ngf=ngf)
        self.DC_F = VDecoder('DC_F', ngf=ngf, output_channl=2)

        self.D_F = Discriminator('D_F', ngf=ngf)
        self.FD_F = FeatureDiscriminator('FD_F', ngf=ngf)

    def get_f(self, x, beta=0.07):
        f1 = self.norm(tf.reduce_min(tf.image.sobel_edges(x), axis=-1))
        f2 = self.norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))
        f1 = tf.reduce_mean(f1, axis=[1, 2, 3]) - f1
        f2 = f2 - tf.reduce_mean(f2, axis=[1, 2, 3])

        f1 = self.ones * tf.cast(f1 > beta, dtype="float32")
        f2 = self.ones * tf.cast(f2 > beta, dtype="float32")

        f = f1 + f2
        f = self.ones * tf.cast(f > 0.0, dtype="float32")
        return f

    def get_mask(self, m, p=5):
        mask = 1.0 - self.ones * tf.cast(m > 0.0, dtype="float32")
        shape = m.get_shape().as_list()
        mask = tf.image.resize_images(mask, size=[shape[1] + p, shape[2] + p], method=1)
        mask = tf.image.resize_image_with_crop_or_pad(mask, shape[1], shape[2])
        return mask

    def remove_l(self, l, f):
        l_mask = self.get_mask(l, p=0)
        f = f * l_mask
        return f

    def model(self, l_m, m):
        mask = self.get_mask(m)
        f = self.get_f(m)  # M->F
        f = self.remove_l(l_m, f)

        # F -> F_R VAE
        code_f_mean, code_f_logvar = self.EC_F(f)
        shape = code_f_logvar.get_shape().as_list()
        code_f_std = tf.exp(0.5 * code_f_logvar)
        code_f_epsilon = tf.random_normal(shape, mean=0., stddev=1., dtype=tf.float32)
        code_f = code_f_mean + tf.multiply(code_f_std, code_f_epsilon)

        f_r_prob = self.DC_F(code_f)
        f_r = tf.reshape(tf.cast(tf.argmax(f_r_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        code_f_mask = self.EC_MASK(f)
        mask_r_prob = self.DC_MASK(code_f_mask)
        mask_r = tf.reshape(tf.cast(tf.argmax(mask_r_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        # CODE_F_RM
        code_f_rm = tf.random_normal(shape, mean=0., stddev=1., dtype=tf.float32)

        f_rm_prob = self.DC_F(code_f_rm)
        f_rm = tf.reshape(tf.cast(tf.argmax(f_rm_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        code_f_rm_mask = self.EC_MASK(f_rm)
        mask_rm_prob = self.DC_MASK(code_f_rm_mask)
        mask_rm = tf.reshape(tf.cast(tf.argmax(mask_rm_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        self.tenaor_name["code_f_rm"] = str(code_f_rm)
        self.tenaor_name["f_rm"] = str(f_rm)
        self.tenaor_name["mask_rm"] = str(mask_rm)

        # D,FD
        j_f = self.D_F(f)
        j_f_rm = self.D_F(f_rm)
        self.tenaor_name["j_f_rm"] = str(j_f_rm)

        code_f = tf.reshape(code_f, shape=[-1, 64, 64, 1])
        code_f_rm = tf.reshape(code_f_rm, shape=[-1, 64, 64, 1])
        j_code_f_rm = self.FD_F(code_f_rm)
        j_code_f = self.FD_F(code_f)

        D_loss = 0.0
        FG_loss = 0.0
        MG_loss = 0.0
        D_loss += self.mse_loss(j_code_f_rm, 1.0) * 0.1
        D_loss += self.mse_loss(j_code_f, 0.0) * 0.1
        FG_loss += self.mse_loss(j_code_f, 1.0) * 0.1

        FG_loss += self.mse_loss(tf.reduce_mean(code_f_mean), 0.0) * 0.1
        FG_loss += self.mse_loss(tf.reduce_mean(code_f_std), 1.0) * 0.1

        D_loss += self.mse_loss(j_f, 1.0) * 0.001
        D_loss += self.mse_loss(j_f_rm, 0.0) * 0.001
        FG_loss += self.mse_loss(j_f_rm, 1.0) * 100

        FG_loss += self.mse_loss(f, f_r) * 50
        MG_loss += self.mse_loss(mask, mask_r) * 25

        FG_loss += self.mse_loss(0.0, f_r * mask) * 5
        MG_loss += self.mse_loss(0.0, f * mask_r) * 5
        MG_loss += self.mse_loss(0.0, f_r * mask_r) * 5
        MG_loss += self.mse_loss(0.0, f_rm * mask_rm) * 5

        f_one_hot = tf.reshape(tf.one_hot(tf.cast(f, dtype=tf.int32), depth=2, axis=-1),
                               shape=f_r_prob.get_shape().as_list()) * 5
        FG_loss += self.mse_loss(f_one_hot, f_r_prob) * 50
        mask_one_hot = tf.reshape(tf.one_hot(tf.cast(mask, dtype=tf.int32), depth=2, axis=-1),
                                  shape=mask_r_prob.get_shape().as_list())
        MG_loss += self.mse_loss(mask_one_hot, mask_r_prob) * 25

        image_list = [m, f, f_r, f_rm, mask, mask_r, mask_rm]

        code_list = [code_f, code_f_rm]

        j_list = [j_code_f, j_code_f_rm, j_f, j_f_rm]

        loss_list = [FG_loss, MG_loss, D_loss]

        return image_list, code_list, j_list, loss_list

    def get_variables(self):
        return [self.EC_F.variables
                + self.DC_F.variables
            ,
                self.EC_MASK.variables
                + self.DC_MASK.variables
            ,
                self.D_F.variables +
                self.FD_F.variables
                ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        FG_optimizer = make_optimizer(name='Adam_FG')
        MG_optimizer = make_optimizer(name='Adam_MG')
        D_optimizer = make_optimizer(name='Adam_D')

        return FG_optimizer, MG_optimizer, D_optimizer

    def mse_loss(self, x, y):
        loss = tf.reduce_mean(tf.square(x - y))
        return loss

    def ssim_loss(self, x, y):
        loss = (1.0 - self.SSIM(x, y)) * 20
        return loss

    def PSNR(self, output, target):
        psnr = tf.reduce_mean(tf.image.psnr(output, target, max_val=1.0, name="psnr"))
        return psnr

    def SSIM(self, output, target):
        ssim = tf.reduce_mean(tf.image.ssim(output, target, max_val=1.0))
        return ssim

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output
