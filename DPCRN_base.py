# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:16:58 2020

@author: Xiaohuai Le
"""
import os
from random import seed

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow.keras as keras
import tf2onnx
from tensorflow.keras.layers import Activation, Lambda, Input, LayerNormalization, Conv2D, BatchNormalization, \
    Conv2DTranspose, Concatenate, PReLU
from tensorflow.keras.models import Model

from loss import Loss
from networks.modules import DprnnBlock
from signal_processing import Signal_Pro

seed(42)
np.random.seed(42)


class DPCRN_model(Loss, Signal_Pro):
    '''
    Class to create the DPCRN-base model
    '''

    def __init__(self, batch_size, config, length_in_s=8, lr=1e-3):
        '''
        Constructor
        '''
        Signal_Pro.__init__(self, config)

        self.network_config = config['network']
        self.filter_size = self.network_config['filter_size']
        self.kernel_size = self.network_config['kernel_size']
        self.strides = self.network_config['strides']
        self.encoder_padding = self.network_config['encoder_padding']
        self.decoder_padding = self.network_config['decoder_padding']
        self.output_cut_off = self.network_config['output_cut']
        self.N_DPRNN = self.network_config['N_DPRNN']
        self.use_CuDNNGRU = self.network_config['use_CuDNNGRU']
        self.activation = self.network_config['activation']
        self.input_norm = self.network_config['input_norm']
        self.intra_hidden_size = self.network_config['DPRNN']['intra_hidden_size']
        self.inter_hidden_size = self.network_config['DPRNN']['inter_hidden_size']
        # empty property for the model
        self.model = None
        # defining default parameters
        self.length_in_s = length_in_s
        self.batch_size = batch_size
        self.lr = lr
        self.eps = 1e-9

        self.L = (16000 * length_in_s - self.block_len) // self.block_shift + 1

    def metricsWrapper(self):
        '''
        A wrapper function which returns the metrics used during training
        '''
        return [self.sisnr_cost]

    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        '''

        def spectrum_loss_SD(s_hat, s, c=0.3, Lam=0.1):
            # The complex compressed spectrum MSE loss
            s = tf.truediv(s, self.batch_gain + 1e-9)
            s_hat = tf.truediv(s_hat, self.batch_gain + 1e-9)

            true_real, true_imag = self.stftLayer(s, mode='real_imag')
            hat_real, hat_imag = self.stftLayer(s_hat, mode='real_imag')

            true_mag = tf.sqrt(true_real ** 2 + true_imag ** 2 + 1e-9)
            hat_mag = tf.sqrt(hat_real ** 2 + hat_imag ** 2 + 1e-9)

            true_real_cprs = (true_real / true_mag) * true_mag ** c
            true_imag_cprs = (true_imag / true_mag) * true_mag ** c
            hat_real_cprs = (hat_real / hat_mag) * hat_mag ** c
            hat_imag_cprs = (hat_imag / hat_mag) * hat_mag ** c

            loss_mag = tf.reduce_mean((hat_mag ** c - true_mag ** c) ** 2, )
            loss_real = tf.reduce_mean((hat_real_cprs - true_real_cprs) ** 2, )
            loss_imag = tf.reduce_mean((hat_imag_cprs - true_imag_cprs) ** 2, )

            return (1 - Lam) * loss_mag + Lam * (loss_imag + loss_real)

        return spectrum_loss_SD

    def build_DPCRN_model(self, name='model0'):

        # input layer for time signal
        time_data = Input(batch_shape=(self.batch_size, None))
        self.batch_gain = Input(batch_shape=(self.batch_size, 1))

        # calculate STFT
        real, imag = Lambda(self.stftLayer, arguments={'mode': 'real_imag'})(time_data)

        real = tf.reshape(real, [self.batch_size, -1, self.block_len // 2 + 1, 1])
        imag = tf.reshape(imag, [self.batch_size, -1, self.block_len // 2 + 1, 1])

        input_mag = tf.math.sqrt(real ** 2 + imag ** 2 + 1e-9)
        input_log_spec = 2 * tf.math.log(input_mag)
        # input feature
        input_complex_spec = Concatenate(axis=-1)([real, imag, input_log_spec])

        '''encoder'''

        if self.input_norm == 'batchnorm':
            input_complex_spec = BatchNormalization(axis=[-1, -2], epsilon=self.eps)(input_complex_spec)
        elif self.input_norm == 'instantlayernorm':
            input_complex_spec = LayerNormalization(axis=[-1, -2], epsilon=self.eps)(input_complex_spec)

        padded_1 = tf.pad(input_complex_spec, [[0, 0], [0, 0], self.encoder_padding[0], [0, 0]], "CONSTANT")

        # conv_1 = Conv2D(self.filter_size[0], self.kernel_size[0], self.strides[0], name=name + '_conv_1',
        #                 padding=[[0, 0], [0, 0], self.encoder_padding[0], [0, 0]])(input_complex_spec)
        conv_1 = Conv2D(self.filter_size[0], self.kernel_size[0], self.strides[0], name=name + '_conv_1')(padded_1)
        bn_1 = BatchNormalization(name=name + '_bn_1')(conv_1)
        out_1 = PReLU(shared_axes=[1, 2])(bn_1)

        padded_2 = tf.pad(out_1, [[0, 0], [0, 0], self.encoder_padding[1], [0, 0]], "CONSTANT")

        # conv_2 = Conv2D(self.filter_size[1], self.kernel_size[1], self.strides[1], name=name + '_conv_2',
        #                 padding=[[0, 0], [0, 0], self.encoder_padding[1], [0, 0]])(out_1)

        conv_2 = Conv2D(self.filter_size[1], self.kernel_size[1], self.strides[1], name=name + '_conv_2')(padded_2)
        bn_2 = BatchNormalization(name=name + '_bn_2')(conv_2)
        out_2 = PReLU(shared_axes=[1, 2])(bn_2)

        padded_3 = tf.pad(out_2, [[0, 0], [0, 0], self.encoder_padding[2], [0, 0]], "CONSTANT")

        # conv_3 = Conv2D(self.filter_size[2], self.kernel_size[2], self.strides[2], name=name + '_conv_3',
        #                 padding=[[0, 0], [0, 0], self.encoder_padding[2], [0, 0]])(out_2)

        conv_3 = Conv2D(self.filter_size[2], self.kernel_size[2], self.strides[2], name=name + '_conv_3')(padded_3)
        bn_3 = BatchNormalization(name=name + '_bn_3')(conv_3)
        out_3 = PReLU(shared_axes=[1, 2])(bn_3)

        padded_4 = tf.pad(out_3, [[0, 0], [0, 0], self.encoder_padding[3], [0, 0]], "CONSTANT")

        # conv_4 = Conv2D(self.filter_size[3], self.kernel_size[3], self.strides[3], name=name + '_conv_4',
        #                 padding=[[0, 0], [0, 0], self.encoder_padding[3], [0, 0]])(out_3)

        conv_4 = Conv2D(self.filter_size[3], self.kernel_size[3], self.strides[3], name=name + '_conv_4')(padded_4)
        bn_4 = BatchNormalization(name=name + '_bn_4')(conv_4)
        out_4 = PReLU(shared_axes=[1, 2])(bn_4)

        padded_5 = tf.pad(out_4, [[0, 0], [0, 0], self.encoder_padding[4], [0, 0]], "CONSTANT")

        # conv_5 = Conv2D(self.filter_size[4], self.kernel_size[4], self.strides[4], name=name + '_conv_5',
        #                 padding=[[0, 0], [0, 0], self.encoder_padding[4], [0, 0]])(out_4)

        conv_5 = Conv2D(self.filter_size[4], self.kernel_size[4], self.strides[4], name=name + '_conv_5')(padded_5)
        bn_5 = BatchNormalization(name=name + '_bn_5')(conv_5)
        out_5 = PReLU(shared_axes=[1, 2])(bn_5)

        dp_in = out_5

        for i in range(self.N_DPRNN):
            dp_in = DprnnBlock(intra_hidden=self.intra_hidden_size,
                               inter_hidden=self.inter_hidden_size,
                               batch_size=self.batch_size,
                               L=-1,
                               width=self.block_len // 2 // 8,
                               channel=self.filter_size[4],
                               causal=True,
                               CUDNN=self.use_CuDNNGRU)(dp_in)

        print(dp_in.shape)
        dp_out = dp_in
        '''decoder'''
        skipcon_1 = Concatenate(axis=-1)([out_5, dp_out])

        deconv_1 = Conv2DTranspose(self.filter_size[3], self.kernel_size[4], self.strides[4], name=name + '_dconv_1',
                                   padding=self.decoder_padding[0])(skipcon_1)
        dbn_1 = BatchNormalization(name=name + '_dbn_1')(deconv_1)
        dout_1 = PReLU(shared_axes=[1, 2])(dbn_1)

        skipcon_2 = Concatenate(axis=-1)([out_4, dout_1])

        deconv_2 = Conv2DTranspose(self.filter_size[2], self.kernel_size[3], self.strides[3], name=name + '_dconv_2',
                                   padding=self.decoder_padding[1])(skipcon_2)
        dbn_2 = BatchNormalization(name=name + '_dbn_2')(deconv_2)
        dout_2 = PReLU(shared_axes=[1, 2])(dbn_2)

        skipcon_3 = Concatenate(axis=-1)([out_3, dout_2])

        deconv_3 = Conv2DTranspose(self.filter_size[1], self.kernel_size[2], self.strides[2], name=name + '_dconv_3',
                                   padding=self.decoder_padding[2])(skipcon_3)
        dbn_3 = BatchNormalization(name=name + '_dbn_3')(deconv_3)
        dout_3 = PReLU(shared_axes=[1, 2])(dbn_3)

        skipcon_4 = Concatenate(axis=-1)([out_2, dout_3])

        deconv_4 = Conv2DTranspose(self.filter_size[0], self.kernel_size[1], self.strides[1], name=name + '_dconv_4',
                                   padding=self.decoder_padding[3])(skipcon_4)
        dbn_4 = BatchNormalization(name=name + '_dbn_4')(deconv_4)
        dout_4 = PReLU(shared_axes=[1, 2])(dbn_4)

        skipcon_5 = Concatenate(axis=-1)([out_1, dout_4])

        deconv_5 = Conv2DTranspose(2, self.kernel_size[0], self.strides[0], name=name + '_dconv_5',
                                   padding=self.decoder_padding[4])(skipcon_5)

        deconv_5 = deconv_5[:, :, :-self.output_cut_off]

        dbn_5 = BatchNormalization(name=name + '_dbn_5')(deconv_5)

        mag_mask = Conv2DTranspose(1, self.kernel_size[0], self.strides[0], name=name + 'mag_mask',
                                   padding=self.decoder_padding[4])(skipcon_5)[:, :, :-self.output_cut_off, 0]

        # get magnitude mask
        if self.activation == 'sigmoid':
            self.mag_mask = Activation('sigmoid')(BatchNormalization()(mag_mask)) * 1.2
        elif self.activation == 'softplus':
            self.mag_mask = Activation('softplus')(BatchNormalization()(mag_mask))
        # get phase mask
        phase_square = tf.math.sqrt(dbn_5[:, :, :, 0] ** 2 + dbn_5[:, :, :, 1] ** 2 + self.eps)

        self.phase_sin = dbn_5[:, :, :, 1] / phase_square
        self.phase_cos = dbn_5[:, :, :, 0] / phase_square

        self.enh_mag_real, self.enh_mag_imag = Lambda(self.mk_mask_mag)([real, imag, self.mag_mask])

        enh_spec = Lambda(self.mk_mask_pha)([self.enh_mag_real, self.enh_mag_imag, self.phase_cos, self.phase_sin])

        enh_frame = Lambda(self.ifftLayer, arguments={'mode': 'real_imag'})(enh_spec)
        enh_frame = enh_frame * self.win
        enh_time = Lambda(self.overlapAddLayer, name='enhanced_time')(enh_frame)

        self.model = Model([time_data, self.batch_gain], enh_time)
        self.model.summary()

        self.model_inference = Model(time_data, enh_time)

        return self.model

    def compile_model(self):
        '''
        Method to compile the model for training
        '''
        # use the Adam optimizer with a clipnorm of 3
        optimizerAdam = keras.optimizers.Adam(lr=self.lr, clipnorm=3.0)
        # compile model with loss function
        self.model.compile(loss=self.lossWrapper(), optimizer=optimizerAdam, metrics=self.metricsWrapper())

    def enhancement(self, noisy_f, output_f='./enhance_s.wav', plot=True, gain=1):

        noisy_s = sf.read(noisy_f, dtype='float32')[0]  # [:400]

        enh_s = self.model_inference.predict(np.array([noisy_s]) * gain)

        enh_s = enh_s[0]

        if plot:
            spec_n = librosa.stft(noisy_s, 512, 256, center=False)
            spec_e = librosa.stft(enh_s, 512, 256, center=False)
            plt.figure(0)
            plt.plot(noisy_s)
            plt.plot(enh_s)
            plt.figure(1)
            plt.subplot(211)
            plt.imshow(np.log(abs(spec_n) + 1e-8), cmap='jet', origin='lower')
            plt.subplot(212)
            plt.imshow(np.log(abs(spec_e) + 1e-8), cmap='jet', origin='lower')
        sf.write(output_f, enh_s, 16000)

        return noisy_s, enh_s

    def test_on_dataset(self, noisy_path, target_path):
        import tqdm
        f_list = os.listdir(noisy_path)
        for f in tqdm.tqdm(f_list):
            self.enhancement(noisy_f=os.path.join(noisy_path, f), output_f=os.path.join(target_path, f), plot=False)

    def DprnnBlock_stateful(self, x, initial_states, numUnits, batch_size, L, width, channel, causal, CUDNN):
        states_h = []

        for idx in range(self.N_DPRNN):
            # tf2.x
            in_state = initial_states[0, idx, :, :]
            intra_LSTM_input = tf.reshape(x, [-1, width, channel])

            intra_LSTM_out = keras.layers.Bidirectional(
                keras.layers.GRU(units=int(numUnits // 2), return_sequences=True, implementation=1,
                                 recurrent_activation='sigmoid', unroll=True, reset_after=False))(intra_LSTM_input)

            intra_dense_out = keras.layers.Dense(units=channel, )(intra_LSTM_out)

            if causal:
                # (bs*T,F,C) --> (bs,T,F,C) Freq and channel norm
                intra_ln_input = tf.reshape(intra_dense_out, [batch_size, -1, width, channel])
                intra_out = keras.layers.LayerNormalization(center=True, scale=True, epsilon=1e-8)(intra_ln_input)

            intra_out = keras.layers.Add()([x, intra_out])

            # (bs,T,F,C) --> (bs,F,T,C)
            inter_LSTM_input = tf.transpose(intra_out, [0, 2, 1, 3])
            # (bs,F,T,C) --> (bs*F,T,C)
            inter_LSTM_input = tf.reshape(inter_LSTM_input, [batch_size * width, L, channel])

            # if idx == 0:
            inter_LSTM_out, inter_LSTM_h = keras.layers.GRU(units=numUnits, return_sequences=True, implementation=1,
                                                            return_state=True, recurrent_activation='sigmoid',
                                                            reset_after=False)(inter_LSTM_input, initial_state=in_state)

            inter_dense_out = keras.layers.Dense(units=channel, )(inter_LSTM_out)

            inter_dense_out = tf.reshape(inter_dense_out, [batch_size, width, L, channel])

            if causal:
                # (bs,F,T,C) --> (bs,T,F,C)
                inter_ln_input = tf.transpose(inter_dense_out, [0, 2, 1, 3])
                inter_out = keras.layers.LayerNormalization(center=True, scale=True, epsilon=1e-8)(inter_ln_input)

            # (bs,T,F,C)
            inter_out = keras.layers.Add()([intra_out, inter_out])

            states_h.append(inter_LSTM_h)
            x = inter_out

        out_states_h = tf.reshape(tf.stack(states_h, axis=0),
                                  [-1, self.N_DPRNN, width, numUnits])
        return inter_out, out_states_h

    def create_tf_lite_model(self, weights_file, target_folder, use_dynamic_range_quant):
        """
        Created on 2022/10/18
        author:
        tf2.3
        """
        name = 'model0'
        self.build_DPCRN_model()
        self.model.load_weights(weights_file)

        input_complex_spec = Input(batch_shape=(1, 1, self.block_len // 2 + 1, 3))

        '''encoder'''

        if self.input_norm == 'batchnorm':
            input_complex_norm = BatchNormalization(axis=[-1, -2], epsilon=self.eps)(input_complex_spec)
        elif self.input_norm == 'instantlayernorm':
            input_complex_norm = LayerNormalization(axis=[-1, -2], epsilon=self.eps)(input_complex_spec)

        padded_1 = tf.pad(input_complex_norm, [[0, 0], [0, 0], self.encoder_padding[0], [0, 0]], "CONSTANT")

        conv_1 = Conv2D(self.filter_size[0], self.kernel_size[0], self.strides[0], name=name + '_conv_1')(padded_1)
        bn_1 = BatchNormalization(name=name + '_bn_1')(conv_1)
        out_1 = PReLU(shared_axes=[1, 2])(bn_1)

        padded_2 = tf.pad(out_1, [[0, 0], [0, 0], self.encoder_padding[1], [0, 0]], "CONSTANT")

        conv_2 = Conv2D(self.filter_size[1], self.kernel_size[1], self.strides[1], name=name + '_conv_2')(padded_2)
        bn_2 = BatchNormalization(name=name + '_bn_2')(conv_2)
        out_2 = PReLU(shared_axes=[1, 2])(bn_2)

        padded_3 = tf.pad(out_2, [[0, 0], [0, 0], self.encoder_padding[2], [0, 0]], "CONSTANT")

        conv_3 = Conv2D(self.filter_size[2], self.kernel_size[2], self.strides[2], name=name + '_conv_3')(padded_3)
        bn_3 = BatchNormalization(name=name + '_bn_3')(conv_3)
        out_3 = PReLU(shared_axes=[1, 2])(bn_3)

        padded_4 = tf.pad(out_3, [[0, 0], [0, 0], self.encoder_padding[3], [0, 0]], "CONSTANT")

        conv_4 = Conv2D(self.filter_size[3], self.kernel_size[3], self.strides[3], name=name + '_conv_4')(padded_4)
        bn_4 = BatchNormalization(name=name + '_bn_4')(conv_4)
        out_4 = PReLU(shared_axes=[1, 2])(bn_4)

        padded_5 = tf.pad(out_4, [[0, 0], [0, 0], self.encoder_padding[4], [0, 0]], "CONSTANT")

        conv_5 = Conv2D(self.filter_size[4], self.kernel_size[4], self.strides[4], name=name + '_conv_5')(padded_5)
        bn_5 = BatchNormalization(name=name + '_bn_5')(conv_5)
        out_5 = PReLU(shared_axes=[1, 2])(bn_5)

        dp_in = out_5

        states_in = Input(batch_shape=(1, 2, self.block_len // 2 // 8, self.inter_hidden_size))

        dp_out_1, states_out = self.DprnnBlock_stateful(dp_in, states_in, self.intra_hidden_size,
                                                        batch_size=self.batch_size,
                                                        L=-1,
                                                        width=self.block_len // 2 // 8,
                                                        channel=self.filter_size[4],
                                                        causal=True,
                                                        CUDNN=self.use_CuDNNGRU)

        # dp_out = dp_in
        dp_out = dp_out_1
        '''decoder'''
        skipcon_1 = Concatenate(axis=-1)([out_5, dp_out])

        deconv_1 = Conv2DTranspose(self.filter_size[3], self.kernel_size[4], self.strides[4], name=name + '_dconv_1',
                                   padding=self.decoder_padding[0])(skipcon_1)
        dbn_1 = BatchNormalization(name=name + '_dbn_1')(deconv_1)
        dout_1 = PReLU(shared_axes=[1, 2])(dbn_1)

        skipcon_2 = Concatenate(axis=-1)([out_4, dout_1])

        deconv_2 = Conv2DTranspose(self.filter_size[2], self.kernel_size[3], self.strides[3], name=name + '_dconv_2',
                                   padding=self.decoder_padding[1])(skipcon_2)
        dbn_2 = BatchNormalization(name=name + '_dbn_2')(deconv_2)
        dout_2 = PReLU(shared_axes=[1, 2])(dbn_2)

        skipcon_3 = Concatenate(axis=-1)([out_3, dout_2])

        deconv_3 = Conv2DTranspose(self.filter_size[1], self.kernel_size[2], self.strides[2], name=name + '_dconv_3',
                                   padding=self.decoder_padding[2])(skipcon_3)
        dbn_3 = BatchNormalization(name=name + '_dbn_3')(deconv_3)
        dout_3 = PReLU(shared_axes=[1, 2])(dbn_3)

        skipcon_4 = Concatenate(axis=-1)([out_2, dout_3])

        deconv_4 = Conv2DTranspose(self.filter_size[0], self.kernel_size[1], self.strides[1], name=name + '_dconv_4',
                                   padding=self.decoder_padding[3])(skipcon_4)
        dbn_4 = BatchNormalization(name=name + '_dbn_4')(deconv_4)
        dout_4 = PReLU(shared_axes=[1, 2])(dbn_4)

        skipcon_5 = Concatenate(axis=-1)([out_1, dout_4])

        deconv_5 = Conv2DTranspose(2, self.kernel_size[0], self.strides[0], name=name + '_dconv_5',
                                   padding=self.decoder_padding[4])(skipcon_5)

        deconv_5 = deconv_5[:, :, :-self.output_cut_off]

        dbn_5 = BatchNormalization(name=name + '_dbn_5')(deconv_5)

        mag_mask = Conv2DTranspose(1, self.kernel_size[0], self.strides[0], name=name + 'mag_mask',
                                   padding=self.decoder_padding[4])(skipcon_5)[:, :, :-self.output_cut_off, 0]

        # get magnitude mask
        if self.activation == 'sigmoid':
            self.mag_mask = Activation('sigmoid')(BatchNormalization()(mag_mask)) * 1.2
        elif self.activation == 'softplus':
            self.mag_mask = Activation('softplus')(BatchNormalization()(mag_mask))

        phase_square = tf.math.sqrt(dbn_5[:, :, :, 0] ** 2 + dbn_5[:, :, :, 1] ** 2 + self.eps)

        self.phase_sin = dbn_5[:, :, :, 1] / phase_square
        self.phase_cos = dbn_5[:, :, :, 0] / phase_square

        output_mask = self.mag_mask

        output_sin = self.phase_sin

        output_cos = self.phase_cos

        model_1 = Model([input_complex_spec, states_in], [output_mask, output_cos, output_sin, states_out])
        model_1.summary()
        weights = self.model.get_weights()
        model_1.set_weights(weights)

        # convert model
        converter = tf.lite.TFLiteConverter.from_keras_model(model_1)
        # converter.optimizations = ["DEFAULT"]
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(target_folder + 'dpcrn_quant.tflite', 'wb') as f:
            f.write(tflite_model)
        pass

    def create_onnx_model(self, weights_file, target_folder, use_dynamic_range_quant):
        """
        Created on 2022/10/18
        author:
        tf2.3
        """
        name = 'model0'
        self.build_DPCRN_model()
        self.model.load_weights(weights_file)

        input_complex_spec = Input(batch_shape=(1, 1, self.block_len // 2 + 1, 3))

        '''encoder'''

        if self.input_norm == 'batchnorm':
            input_complex_norm = BatchNormalization(axis=[-1, -2], epsilon=self.eps)(input_complex_spec)
        elif self.input_norm == 'instantlayernorm':
            input_complex_norm = LayerNormalization(axis=[-1, -2], epsilon=self.eps)(input_complex_spec)

        padded_1 = tf.pad(input_complex_norm, [[0, 0], [0, 0], self.encoder_padding[0], [0, 0]], "CONSTANT")

        conv_1 = Conv2D(self.filter_size[0], self.kernel_size[0], self.strides[0], name=name + '_conv_1')(padded_1)
        bn_1 = BatchNormalization(name=name + '_bn_1')(conv_1)
        out_1 = PReLU(shared_axes=[1, 2])(bn_1)

        padded_2 = tf.pad(out_1, [[0, 0], [0, 0], self.encoder_padding[1], [0, 0]], "CONSTANT")

        conv_2 = Conv2D(self.filter_size[1], self.kernel_size[1], self.strides[1], name=name + '_conv_2')(padded_2)
        bn_2 = BatchNormalization(name=name + '_bn_2')(conv_2)
        out_2 = PReLU(shared_axes=[1, 2])(bn_2)

        padded_3 = tf.pad(out_2, [[0, 0], [0, 0], self.encoder_padding[2], [0, 0]], "CONSTANT")

        conv_3 = Conv2D(self.filter_size[2], self.kernel_size[2], self.strides[2], name=name + '_conv_3')(padded_3)
        bn_3 = BatchNormalization(name=name + '_bn_3')(conv_3)
        out_3 = PReLU(shared_axes=[1, 2])(bn_3)

        padded_4 = tf.pad(out_3, [[0, 0], [0, 0], self.encoder_padding[3], [0, 0]], "CONSTANT")

        conv_4 = Conv2D(self.filter_size[3], self.kernel_size[3], self.strides[3], name=name + '_conv_4')(padded_4)
        bn_4 = BatchNormalization(name=name + '_bn_4')(conv_4)
        out_4 = PReLU(shared_axes=[1, 2])(bn_4)

        padded_5 = tf.pad(out_4, [[0, 0], [0, 0], self.encoder_padding[4], [0, 0]], "CONSTANT")

        conv_5 = Conv2D(self.filter_size[4], self.kernel_size[4], self.strides[4], name=name + '_conv_5')(padded_5)
        bn_5 = BatchNormalization(name=name + '_bn_5')(conv_5)
        out_5 = PReLU(shared_axes=[1, 2])(bn_5)

        dp_in = out_5

        states_in = Input(batch_shape=(1, 2, self.block_len // 2 // 8, self.inter_hidden_size))

        dp_out_1, states_out = self.DprnnBlock_stateful(dp_in, states_in, self.intra_hidden_size,
                                                        batch_size=self.batch_size,
                                                        L=-1,
                                                        width=self.block_len // 2 // 8,
                                                        channel=self.filter_size[4],
                                                        causal=True,
                                                        CUDNN=self.use_CuDNNGRU)

        # dp_out = dp_in
        dp_out = dp_out_1
        '''decoder'''
        skipcon_1 = Concatenate(axis=-1)([out_5, dp_out])

        deconv_1 = Conv2DTranspose(self.filter_size[3], self.kernel_size[4], self.strides[4], name=name + '_dconv_1',
                                   padding=self.decoder_padding[0])(skipcon_1)
        dbn_1 = BatchNormalization(name=name + '_dbn_1')(deconv_1)
        dout_1 = PReLU(shared_axes=[1, 2])(dbn_1)

        skipcon_2 = Concatenate(axis=-1)([out_4, dout_1])

        deconv_2 = Conv2DTranspose(self.filter_size[2], self.kernel_size[3], self.strides[3], name=name + '_dconv_2',
                                   padding=self.decoder_padding[1])(skipcon_2)
        dbn_2 = BatchNormalization(name=name + '_dbn_2')(deconv_2)
        dout_2 = PReLU(shared_axes=[1, 2])(dbn_2)

        skipcon_3 = Concatenate(axis=-1)([out_3, dout_2])

        deconv_3 = Conv2DTranspose(self.filter_size[1], self.kernel_size[2], self.strides[2], name=name + '_dconv_3',
                                   padding=self.decoder_padding[2])(skipcon_3)
        dbn_3 = BatchNormalization(name=name + '_dbn_3')(deconv_3)
        dout_3 = PReLU(shared_axes=[1, 2])(dbn_3)

        skipcon_4 = Concatenate(axis=-1)([out_2, dout_3])

        deconv_4 = Conv2DTranspose(self.filter_size[0], self.kernel_size[1], self.strides[1], name=name + '_dconv_4',
                                   padding=self.decoder_padding[3])(skipcon_4)
        dbn_4 = BatchNormalization(name=name + '_dbn_4')(deconv_4)
        dout_4 = PReLU(shared_axes=[1, 2])(dbn_4)

        skipcon_5 = Concatenate(axis=-1)([out_1, dout_4])

        deconv_5 = Conv2DTranspose(2, self.kernel_size[0], self.strides[0], name=name + '_dconv_5',
                                   padding=self.decoder_padding[4])(skipcon_5)

        deconv_5 = deconv_5[:, :, :-self.output_cut_off]

        dbn_5 = BatchNormalization(name=name + '_dbn_5')(deconv_5)

        mag_mask = Conv2DTranspose(1, self.kernel_size[0], self.strides[0], name=name + 'mag_mask',
                                   padding=self.decoder_padding[4])(skipcon_5)[:, :, :-self.output_cut_off, 0]

        # get magnitude mask
        if self.activation == 'sigmoid':
            self.mag_mask = Activation('sigmoid')(BatchNormalization()(mag_mask)) * 1.2
        elif self.activation == 'softplus':
            self.mag_mask = Activation('softplus')(BatchNormalization()(mag_mask))

        phase_square = tf.math.sqrt(dbn_5[:, :, :, 0] ** 2 + dbn_5[:, :, :, 1] ** 2 + self.eps)

        self.phase_sin = dbn_5[:, :, :, 1] / phase_square
        self.phase_cos = dbn_5[:, :, :, 0] / phase_square

        output_mask = self.mag_mask

        output_sin = self.phase_sin

        output_cos = self.phase_cos

        model_1 = Model([input_complex_spec, states_in], [output_mask, output_cos, output_sin, states_out])
        model_1.summary()
        weights = self.model.get_weights()
        model_1.set_weights(weights)

        temp_model_file = target_folder + 'dpcrn.onnx'
        # keras2onnx.save_model(onnx_model, temp_model_file)
        (onnx_model_proto, storage) = tf2onnx.convert.from_keras(model_1)
        with open(temp_model_file, "wb") as f:
            f.write(onnx_model_proto.SerializeToString())

        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import yaml

    f = open('./configuration/DPCRN-base.yaml', 'r', encoding='utf-8')
    result = f.read()
    print(result)

    config_dict = yaml.safe_load(result)
    model = DPCRN_model(batch_size=1, length_in_s=5, lr=1e-3, config=config_dict)

    model.build_DPCRN_model()
