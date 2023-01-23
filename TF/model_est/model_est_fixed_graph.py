import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')
import os

import tensorflow as tf
from tensorflow_core.contrib.rnn.python.ops.rnn_cell import WeightNormLSTMCell
from gym.utils import seeding

MODE_Z = 1
MODE_Z_HIDDEN = 2

class missVAE(object):
    def __init__(self, hps, scope, gpu_mode=True, reuse=False):
        self.hps = hps
        self.scope = scope
        self._seed()

        self.input_x = None
        self.input_sign = None
        self.seq_length = None

        self.mu = None
        self.logvar = None

        self.z_s = None
        self.z_r = None
        self.z_map_s = None
        self.z_map_r = None
        self.y_o = None  # VAE output
        self.y_o_next = None
        self.y_r = None

        self.input_z_s = None
        self.input_z_r = None
        self.output_z_s = None
        self.output_z_r = None
        self.input_a = None  # RNN input placeholder for action
        self.input_a_prev = None
        self.input_r = None  # RNN input placeholder for reward
        self.input_r_prev = None
        self.input_r_next = None
        self.cell = None

        self.initial_state = None
        self.final_state_r = None
        self.final_state_s = None

        self.out_logmix = None
        self.out_mean = None
        self.out_logstd = None
        self.z_out_logmix = None
        self.z_out_mean = None
        self.z_out_logstd = None

        self.global_step = None
        self.r_obs_loss = None  # reconstruction loss for observation
        self.r_next_obs_loss = None
        self.r_reward_loss = None  # reconstruction loss for reward
        self.kl_loss = None
        self.vae_loss = None
        self.vae_pure_loss = None
        self.transition_loss = None
        self.vae_causal_filter_loss = None
        self.causal_filter_loss = None
        self.total_loss = None

        self.lr = None
        self.train_op = None
        self.vae_lr = None
        self.vae_train_op = None
        self.transition_lr = None
        self.transition_train_op = None

        self.init = None
        self.assign_ops = None
        self.weightnorm_ops = None
        self.sess = None

        self.merged = None

        self.SSL_A = None  # parameter C_{z^s->o} 
        self.SSL_B = None  # parameter C_{z^r->r} 
        self.SSL_C = None  # parameter C_{a->r} 
        self.SSL_D1 = None  # parameter C_{z^s->z^s} 
        self.SSL_D2 = None  # parameter C_{z^r->z^r} 
        self.SSL_E = None  # parameter C_{a->o} 
        self.SSL_F = None  # parameter C_{o->o} 
        self.SSL_G = None # Parameter C_{o->r}

        if self.hps.is_training == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            self.gpu_config = tf.ConfigProto(device_count={'GPU': 4}, allow_soft_placement=True, log_device_placement=False)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.gpu_config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True, log_device_placement=False)

        self.gpu_config.gpu_options.allow_growth = True

        with tf.variable_scope(self.scope, reuse=reuse):
            if not gpu_mode:
                print("model using cpu")
                self.g = tf.Graph()
                with self.g.as_default():
                    self.build_model()
            else:
                print("model using gpu")
                self.g = tf.Graph()
                with self.g.as_default():
                    self.build_model()
        self.init_session()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def build_model(self):        
        self.seq_length = tf.placeholder(tf.int32)
        self.input_sign = tf.placeholder(tf.int32)

        ############################################# Encoder ##########################################################
        #################################### q(s_t | o_{<=t}, a_{<t}, r_{<t}) ##########################################
        # input of VAE
        self.input_x = tf.placeholder(tf.float32, shape=[self.hps.batch_size, None, 128, 128, 1])
        self.input_a_prev = tf.placeholder(dtype=tf.float32,
                                            shape=[self.hps.batch_size, None, self.hps.action_size])
        self.input_a = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, None, self.hps.action_size])
        self.input_r_prev = tf.placeholder(dtype=tf.float32,
                                            shape=[self.hps.batch_size, None, self.hps.reward_size])
        self.input_r_next = tf.placeholder(dtype=tf.float32,
                                            shape=[self.hps.batch_size, None, self.hps.reward_size])                                  
        self.input_r = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, None, self.hps.reward_size])


        # Encoder: s_t = q(o_t, a_{t-1}, r_{t-1})
        obs_x = tf.reshape(self.input_x, [-1])
        obs_x_next = tf.reshape(self.input_x[:, 1:, :, :, :], [-1])
        obs_a_prev = tf.reshape(self.input_a_prev, [-1, self.hps.action_size])
        obs_a = tf.reshape(self.input_a, [-1, self.hps.action_size])
        obs_r_prev = tf.reshape(self.input_r_prev, [-1, self.hps.reward_size])
        obs_r_next = tf.reshape(self.input_r_next, [-1, self.hps.reward_size])
        obs_r = tf.reshape(self.input_r, [-1, self.hps.reward_size])

        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            # obs_x
            hx = tf.layers.dense(obs_x, units=128, activation=tf.nn.relu)
            hx = tf.layers.dense(hx, units=128, activation=tf.nn.relu)
            hx = tf.layers.dense(hx, units=64, activation=tf.nn.relu)

            # obs_a
            ha = tf.layers.dense(obs_a_prev, 1 * 128, activation=tf.nn.relu, name="enc_action_fc1")

            # obs_r
            hr = tf.layers.dense(obs_r_prev, 1 * 128, activation=tf.nn.relu, name="enc_reward_fc1")
            
            h_xa = tf.concat([hx, ha], 1)
            h_xar = tf.concat([hx, ha, hr], 1)
            
            ################################################### LSTM ###################################################
            input_s = tf.reshape(h_xa, [self.hps.batch_size, self.seq_length, 6 * 6 * 256 + 1 * 128])
            input_r = tf.reshape(h_xar, [self.hps.batch_size, self.seq_length, 6 * 6 * 256 + 2 * 128])

            cell_s = WeightNormLSTMCell(self.hps.rnn_size_s, norm=True)
            self.cell_s = cell_s

            self.initial_state_s = cell.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32)

            with tf.variable_scope('RNN-s'):
                output_w_s = tf.get_variable("output_w_s", [self.hps.rnn_size_s, NOUT],
                                            initializer=tf.contrib.layers.xavier_initializer())
                output_w_s = tf.nn.l2_normalize(output_w_s, [0])
                output_b_s = tf.get_variable("output_b_s", [NOUT],
                                            initializer=tf.contrib.layers.xavier_initializer())

            output_s, last_state_s = tf.nn.dynamic_rnn(cell_s, input_s, initial_state=self.initial_state_s,
                                                    time_major=False, swap_memory=True, dtype=tf.float32,
                                                    scope="RNN-s")

            output_s = tf.reshape(output_s, [-1, self.hps.rnn_size_s])
            output_s = tf.nn.xw_plus_b(output_s, output_w_s, output_b_s)
            output_s = tf.reshape(output_s, [-1, self.hps.num_mixture * 3])
            self.final_state_s = last_state_s
            cell_r = WeightNormLSTMCell(self.hps.rnn_rize_r, norm=True)
            self.cell_r = cell_r

            self.initial_rtate_r = cell.zero_rtate(batch_rize=self.hps.batch_rize, dtype=tf.float32)

            with tf.variable_rcope('RNN-s'):
                output_w_r = tf.get_variable("output_w_r", [self.hps.rnn_rize_r, NOUT],
                                            initializer=tf.contrib.layers.xavier_initializer())
                output_w_r = tf.nn.l2_normalize(output_w_r, [0])
                output_b_r = tf.get_variable("output_b_r", [NOUT],
                                            initializer=tf.contrib.layers.xavier_initializer())

            output_r, last_rtate_r = tf.nn.dynamic_rnn(cell_r, input_r, initial_rtate=self.initial_rtate_r,
                                                    time_major=False, swap_memory=True, dtype=tf.float32,
                                                    scope="RNN-s")

            output_r = tf.reshape(output_r, [-1, self.hps.rnn_rize_r])
            output_r = tf.nn.xw_plus_b(output_r, output_w_r, output_b_r)
            output_r = tf.reshape(output_r, [-1, self.hps.num_mixture * 3])
            self.final_rtate_r = last_rtate_r
            ########################################## MDN-RNN ################################################
            out_logmix_s, out_mean_s, out_logstd_s = self.get_mdn_coef(output_s)
            self.out_logmix_s = out_logmix_s
            self.out_mean_s = out_mean_s
            self.out_logstd_s = out_logstd_s

            if self.hps.is_training == 0:
                # the index of the cluster which has the largest probability
                logmix_map_idx_s = tf.argmax(out_logmix_s, 1)
                out_mean_map_s = []
                for i in range(out_logmix_s.shape[0]):
                    out_mean_map_s.append(out_mean_s[i, logmix_map_idx_s[i]])
                out_mean_map_s = tf.convert_to_tensor(out_mean_map_s)
                self.z_map_s = tf.reshape(out_mean_map_s, [-1, self.hps.output_seq_width])

            logmix2_s = out_logmix_s / self.hps.temperature
            logmix2_s -= tf.reduce_max(logmix2_s)
            logmix2_s = tf.exp(logmix2_s)
            logmix2_s /= tf.reshape(tf.reduce_sum(logmix2_s, 1), [-1, 1])

            mixture_len = self.hps.batch_size * self.seq_length * self.hps.output_seq_width

            out_logmix_r, out_mean_r, out_logstd_r = self.get_mdn_coef(output_r)
            self.out_logmix_r = out_logmix_r
            self.out_mean_r = out_mean_r
            self.out_logstd_r = out_logstd_r
            
            if self.hps.is_training == 0:
                # the index of the cluster which has the largest probability
                logmix_map_idx_r = tf.argmax(out_logmix_r, 1)
                out_mean_map_r = []
                for i in range(out_logmix_r.shape[0]):
                    out_mean_map_r.append(out_mean_r[i, logmix_map_idx_r[i]])
                out_mean_map_r = tf.convert_to_tensor(out_mean_map_r)
                self.z_map_r = tf.reshape(out_mean_map_r, [-1, self.hps.output_req_width])

            logmix2_r = out_logmix_r / self.hps.temperature
            logmix2_r -= tf.reduce_max(logmix2_r)
            logmix2_r = tf.exp(logmix2_r)
            logmix2_r /= tf.reshape(tf.reduce_rum(logmix2_r, 1), [-1, 1])

            ########################################## Sampling from MDN-RNN ###########################################
            logmix2_list_s = [logmix2_s[:, 0]]
            for j in range(self.hps.num_mixture - 1):
                logmix2_list_s.append(logmix2_s[:, j + 1] + logmix2_list_s[j])

            logmix2_s = tf.stack(logmix2_list_s, axis=1)

            mixture_rand_idx = tf.tile(tf.random_uniform([mixture_len, 1]), [1, self.hps.num_mixture])
            zero_ref = tf.zeros_like(mixture_rand_idx)

            idx_s = tf.argmax(tf.cast(tf.less_equal(mixture_rand_idx - logmix2_s, zero_ref), tf.int32),
                            axis=1, output_type=tf.int32)

            indices_s = tf.range(0, mixture_len) * self.hps.num_mixture + idx_s
            chosen_mean_s = tf.gather(tf.reshape(out_mean_s, [-1]), indices_s)
            chosen_logstd_s = tf.gather(tf.reshape(out_logstd_s, [-1]), indices_s)

            rand_gaussian = tf.random_normal([mixture_len]) * np.sqrt(self.hps.temperature)
            sample_z_s = chosen_mean_s + tf.exp(chosen_logstd_s) * rand_gaussian

            self.z_s = tf.reshape(sample_z_s, [-1, self.hps.output_seq_width])

            logmix2_list_r = [logmix2_r[:, 0]]
            for j in range(self.hps.num_mixture - 1):
                logmix2_list_r.append(logmix2_r[:, j + 1] + logmix2_list_r[j])

            logmix2_r = tf.stack(logmix2_list_r, axis=1)

            idx_r = tf.argmax(tf.cast(tf.less_equal(mixture_rand_idx - logmix2_r, zero_ref), tf.int32),
                            axis=1, output_type=tf.int32)

            indices_r = tf.range(0, mixture_len) * self.hps.num_mixture + idx_r
            chosen_mean_r = tf.gather(tf.reshape(out_mean_r, [-1]), indices_r)
            chosen_logstd_r = tf.gather(tf.reshape(out_logstd_r, [-1]), indices_r)

            rand_gaussian = tf.random_normal([mixture_len]) * np.sqrt(self.hps.temperature)
            sample_z_r = chosen_mean_r + tf.exp(chosen_logstd_r) * rand_gaussian

            self.z_r = tf.reshape(sample_z_r, [-1, self.hps.output_req_width])

        ############################################# Decoder ##########################################################
        ssl_zo = tf.multiply(self.z_s, self.SSL_A)  
        ssl_ao = tf.multiply(self.input_a, self.SSL_E)

        with tf.variable_scope('ObsDecoder', reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(tf.concat([ssl_zo, ssl_ao], 1), 128, kernel_constraint=self.deconv_weightnorm, name="dec_fc1")
            self.y_o = tf.layers.dense(h, self.hps.o_size, kernel_constraint=self.deconv_weightnorm, name="dec_fc2")

        # Decoder for Next Observation
        z_s_next = tf.reshape(self.z_s, [self.hps.batch_size, self.seq_length, self.hps.output_seq_width])

        if self.hps.is_training == 1:
            z_s_next = tf.reshape(z_s_next[:, :-1, :], [-1, self.hps.output_seq_width])
        else:
            z_s_next = tf.reshape(z_s_next, [-1, self.hps.output_seq_width])


        with tf.variable_scope('NextObsDecoder', reuse=tf.AUTO_REUSE):
            nh_s = tf.layers.dense(tf.concat([obs_a, obs_x], 1), 128, kernel_constraint=self.deconv_weightnorm, name="dec_fc1")
            nh_s = tf.layers.dense(nh_s, self.hps.o_size, kernel_constraint=self.deconv_weightnorm, name="dec_fc2")
            nh_s = tf.reshape(nh_s, [-1, 1, 1, 6 * 6 * 256])             

        ssl_zr = tf.multiply(self.z_r, self.SSL_B)  
        ssl_ar = tf.multiply(obs_a, self.SSL_C) 
        ssl_or = tf.multiply(obs_x, self.SSL_G)
        ssl_zar = tf.concat([ssl_zr, ssl_ar, ssl_or], 1)

        with tf.variable_scope('RewardDecoder', reuse=tf.AUTO_REUSE):
            lin_h = tf.layers.dense(ssl_zar, 4 * 128, activation=tf.nn.relu,
                                        kernel_constraint=self.deconv_weightnorm, name="dec_fc1")
            lin_h = tf.layers.dense(lin_h, 1 * 128, activation=tf.nn.relu,
                                    kernel_constraint=self.deconv_weightnorm, name="dec_fc2")
            self.y_r = tf.layers.dense(lin_h, 1, kernel_constraint=self.deconv_weightnorm, name="dec_fc3")
        
        ssl_rf = tf.concat([obs_x_next, obs_a, self.z_r])
        with tf.variable_scope('NextRewardDecoder', reuse=tf.AUTO_REUSE):
            lin_h_next = tf.layers.dense(ssl_rf, 4 * 128, activation=tf.nn.relu,
                                        kernel_constraint=self.deconv_weightnorm, name="dec_fc1")
            lin_h_next = tf.layers.dense(lin_h_next, 1 * 128, activation=tf.nn.relu,
                                    kernel_constraint=self.deconv_weightnorm, name="dec_fc2")
            self.y_r_next = tf.layers.dense(lin_h_next, 1, kernel_constraint=self.deconv_weightnorm, name="dec_fc3")

        lossfunc = self.markovian_tran()
        ######################################## Loss Function #########################################################
        if self.hps.is_training == 1:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # VAE Loss
            # reconstruction loss for observation
            self.r_obs_loss = tf.reduce_sum(
                tf.square(obs_x - self.y_o),
                reduction_indices=[1, 2, 3]
            )
            self.r_obs_loss = tf.reduce_mean(self.r_obs_loss)
            # reconstruction loss for next observation
            self.r_next_obs_loss = tf.reduce_sum(
                tf.square(obs_x_next - self.y_o_next),
                reduction_indices=[1, 2, 3]
            )
            self.r_next_obs_loss = tf.reduce_mean(self.r_next_obs_loss)
            # reconstruction loss for reward
            self.r_reward_loss = tf.reduce_sum(
                tf.square(obs_r - self.y_r),
                reduction_indices=[1]
            )
            self.r_reward_next_loss = tf.reduce_sum(
                tf.square(obs_r_next - self.y_r_next),
                reduction_indices=[1]
            )
            self.r_reward_loss = tf.reduce_mean(self.r_reward_loss)
            # KL loss
            self.out_logmix = out_logmix
            self.out_mean = out_mean
            self.out_logstd = out_logstd
            self.kl_loss = 0
            for g_idx in range(self.hps.num_mixture):
                g_logmix = tf.reshape(self.out_logmix[:, g_idx], [-1, self.hps.output_seq_width])
                g_mean = tf.reshape(self.out_mean[:, g_idx], [-1, self.hps.output_seq_width])
                g_logstd = tf.reshape(self.out_logstd[:, g_idx], [-1, self.hps.output_seq_width])
                self.kl_loss += self.kl_gmm(g_logmix, g_mean, g_logstd)
            self.kl_loss = tf.log(1 / (self.kl_loss + 1e-10) + 1e-10)
            self.kl_loss = tf.reduce_mean(self.kl_loss)

            # VAE Causal Filter L1 Loss for sparse constraint
            self.vae_causal_filter_loss = \
                0.1 * tf.reduce_mean(tf.abs(self.SSL_A)) + \
                0.1 * tf.reduce_mean(tf.abs(self.SSL_B)) + \
                0.1 * tf.reduce_mean(tf.abs(self.SSL_C))

            self.vae_loss = self.r_obs_loss + self.r_next_obs_loss + self.r_reward_loss + self.kl_loss + \
                            self.vae_causal_filter_loss
            self.vae_pure_loss = self.r_obs_loss + self.r_next_obs_loss + self.r_reward_loss + self.kl_loss 

            
            self.smooth_loss = tf.reduce_sum(tf.abs(self.z_s[1:] - self.z_s[:-1])) + \
                tf.reduce_sum(tf.abs(self.z_r[1:] - self.z_r[:-1]))


            self.transition_loss = tf.reduce_mean(lossfunc)
            # Causal Filter L1 Loss for sparse constraint
            self.causal_filter_loss = \
                0.1 * tf.reduce_mean(tf.abs(self.SSL_A)) + \
                0.1 * tf.reduce_mean(tf.abs(self.SSL_B)) + \
                0.1 * tf.reduce_mean(tf.abs(self.SSL_C)) + \
                .1 * (tf.reduce_sum(tf.abs(self.SSL_D1)) - tf.reduce_sum(tf.abs(tf.matrix_diag_part(self.SSL_D1)))) \
                / (self.hps.z_s_size * self.hps.z_s_size - self.hps.z_s_size) + \
                0.1 * tf.reduce_mean(tf.abs(tf.matrix_diag_part(self.SSL_D1))) + \
                .1 * (tf.reduce_sum(tf.abs(self.SSL_D2)) - tf.reduce_sum(tf.abs(tf.matrix_diag_part(self.SSL_D2)))) \
                / (self.hps.z_r_size * self.hps.z_r_size - self.hps.z_r_size) + \
                0.1 * tf.reduce_mean(tf.abs(tf.matrix_diag_part(self.SSL_D2))) + \
                0.1 * tf.reduce_mean(tf.abs(self.SSL_E)) +\
                .1 * tf.reduce_mean(tf.abs(self.SSL_G))

            self.total_loss = self.vae_pure_loss + self.transition_loss + self.causal_filter_loss + .02 * self.smooth_loss
            ############################################## Three Optimizers ##########################################            
            self.vae_lr = tf.Variable(self.hps.learning_rate, trainable=False)
            # VAE Optimizer
            vae_optimizer = tf.train.AdamOptimizer(self.vae_lr)
            vae_gvs = vae_optimizer.compute_gradients(self.vae_loss, colocate_gradients_with_ops=True)
            capped_vae_gvs = []
            
            self.vae_train_op = vae_optimizer.apply_gradients(capped_vae_gvs,
                                                                global_step=self.global_step,
                                                                name='vae_train_step')
            # RNN Optimizer
            self.transition_lr = tf.Variable(self.hps.learning_rate, trainable=False)
            transition_optimizer = tf.train.AdamOptimizer(self.transition_lr)

            transition_gvs = transition_optimizer.compute_gradients(self.transition_loss)
            self.transition_train_op = transition_optimizer.apply_gradients(transition_gvs,
                                                                            name='rnn_train_step')
            # Total Optimizer
            self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)

            gvs = optimizer.compute_gradients(self.total_loss)
            capped_gvs = []
            for grad, var in gvs:
                tf.summary.histogram("%s-grad" % var.name, grad)

                def f1(): return grad + tf.random_normal(tf.shape(grad, name=None))

                def f2(): return grad

                grad_tmp = tf.case([(tf.reduce_mean(grad) > 0.01, f2), (tf.equal(self.input_sign, 0), f2)],
                                    default=f1)
                capped_gvs.append((tf.clip_by_value(grad_tmp, -self.hps.grad_clip, self.hps.grad_clip), var))
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step,
                                                        name='train_step')

        # initialize vars
        self.init = tf.global_variables_initializer()
        self.merged = tf.summary.merge_all()
        t_vars = tf.trainable_variables()
        self.assign_ops = {}
        self.weightnorm_ops = {}

        for var in t_vars:
            pshape = var.get_shape()

            pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + '_placeholder')
            assign_op = var.assign(pl)
            self.assign_ops[var] = (assign_op, pl)

            weightnorm_op = var.assign(pl)

            self.weightnorm_ops[var] = (weightnorm_op, pl)

    # Markovian Transition
    def markovian_tran(self):
        D_NOUT = 1 * self.hps.num_mixture * 3

        if self.hps.is_training == 1:
            ssl_zszs = [] 
            for i in range(self.hps.z_s_size):
                ssl_zszs.append(tf.multiply(self.input_z_s, self.SSL_D1[:, i]))
            ssl_zszs = tf.convert_to_tensor(ssl_zszs)  
            ssl_zszs = tf.reshape(ssl_zszs, [-1, self.hps.z_s_size])  

            ssl_zrzr = [] 
            for i in range(self.hps.z_r_size):
                ssl_zrzr.append(tf.multiply(self.input_z_s, self.SSL_D2[:, i]))
            ssl_zrzr = tf.convert_to_tensor(ssl_zrzr)  
            ssl_zrzr = tf.reshape(ssl_zrzr, [-1, self.hps.z_r_size])  

            # SSL for action to state
            ssl_ao = [] 
            for i in range(self.hps.o_size):
                ssl_ao.append(tf.multiply(self.input_a, self.SSL_E[:, i]))
            ssl_ao = tf.convert_to_tensor(ssl_ao)  
            ssl_ao = tf.reshape(ssl_ao, [-1, self.hps.action_size])  
            
            ssl_za = tf.reshape(tf.concat([ssl_zszs, ssl_ao, ssl_zrzr], 1),
                                [-1, self.hps.z_s_size + self.hps.z_r_size + self.hps.action_size])
            
            random_ssl_za = ssl_za
            random_output_z_s = self.output_z_s
        else:
            self.input_z_s = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, None, self.hps.z_s_size])
            tmp_input_z_s = tf.reshape(self.input_z_s, [-1, self.hps.z_s_size])
            # SSL for state to state
            ssl_zz = []  
            for i in range(self.hps.z_size):
                ssl_zz.append(tf.multiply(tmp_input_z_s, self.SSL_D[:, i]))
            ssl_zz = tf.convert_to_tensor(ssl_zz)  
            ssl_zz = tf.reshape(ssl_zz, [-1, self.hps.z_s_size])  # ssl_zz: (z_size x batch_size) x z_size

            tmp_input_a = tf.reshape(self.input_a, [-1, self.hps.action_size])
            # SSL for action to state
            ssl_az = []  
            for i in range(self.hps.z_size):
                ssl_az.append(tf.multiply(tmp_input_a, self.SSL_E[:, i]))
            ssl_az = tf.convert_to_tensor(ssl_az)  # ssl_zz: z_size x batch_size x action_size
            ssl_az = tf.reshape(ssl_az, [-1, self.hps.action_size])  # ssl_zz: (z_size x batch_size) x action_size
            ssl_za = tf.reshape(tf.concat([ssl_zz, ssl_az], 1),
                                [-1, self.hps.z_s_size + self.hps.action_size])
            random_ssl_za = ssl_za

        with tf.variable_scope('Dynamics', reuse=tf.AUTO_REUSE):
            hd = tf.layers.dense(random_ssl_za, 6 * 128, activation=tf.nn.relu,
                                 kernel_constraint=self.deconv_weightnorm, name="fc")
            hd = tf.layers.dense(hd, 4 * 128, activation=tf.nn.relu,
                                 kernel_constraint=self.deconv_weightnorm, name="fc11")
            hd = tf.layers.dense(hd, 2 * 128, activation=tf.nn.relu,
                                 kernel_constraint=self.deconv_weightnorm, name="fc2")
            hd = tf.layers.dense(hd, 128, activation=tf.nn.relu,
                                 kernel_constraint=self.deconv_weightnorm, name="fc3")

            d_output_w = tf.get_variable("output_w", [128, D_NOUT])
            d_output_w = tf.nn.l2_normalize(d_output_w, [0])
            d_output_b = tf.get_variable("output_b", [D_NOUT])

        z_output = tf.nn.xw_plus_b(hd, d_output_w, d_output_b)
        z_output = tf.reshape(z_output, [self.hps.z_size, -1, D_NOUT])
        z_output = tf.transpose(z_output, perm=[1, 0, 2])
        z_output = tf.reshape(z_output, [-1, self.hps.num_mixture * 3])

        z_out_logmix, z_out_mean, z_out_logstd = self.get_mdn_coef(z_output)

        self.z_out_logmix = z_out_logmix
        self.z_out_mean = z_out_mean
        self.z_out_logstd = z_out_logstd

        if self.hps.is_training == 1:
            # reshape target data so that it is compatible with prediction shape
            z_flat_target_data = tf.reshape(random_output_z, [-1, 1])

            lossfunc = self.get_lossfunc(z_out_logmix, z_out_mean, z_out_logstd, z_flat_target_data)
            return lossfunc

    def init_session(self):
        self.sess = tf.Session(graph=self.g, config=self.gpu_config)
        self.sess.run(self.init)

    def reset(self):
        state_init = self.sess.run(self.initial_state)
        action_init = np.zeros((self.hps.batch_size, 1, self.hps.action_size))
        reward_init = np.zeros((self.hps.batch_size, 1, self.hps.reward_size))
        return action_init, reward_init, state_init

    def close_sess(self):
        self.sess.close()

    def encode_s(self, x, a_prev, state_prev=None, seq_len=1):
        seq_len = np.int32(seq_len)
        if state_prev is None:
            state_prev = self.sess.run(self.initial_state)
        cwm_vae_feed = {self.input_x: x,
                        self.input_a_prev: a_prev,
                        self.initial_state: state_prev,
                        self.seq_length: seq_len}
        (z_s, final_state_s) = self.sess.run([self.z_s, self.final_state_s], feed_dict=cwm_vae_feed)
        return z_s, final_state_s
    
    def encode_r(self, x, a_prev, r_prev, state_prev=None, seq_len=1):        
        seq_len = np.int32(seq_len)
        if state_prev is None:
            state_prev = self.sess.run(self.initial_state)
        cwm_vae_feed = {self.input_x: x,
                        self.input_a_prev: a_prev,
                        self.input_r_prev: r_prev,
                        self.initial_state: state_prev,
                        self.seq_length: seq_len}
        (z_r, final_state_r) = self.sess.run([self.z_r, self.final_state_r], feed_dict=cwm_vae_feed)
        return z_r, final_state_r
    
    def encode_new_s(self, x, a_prev, state_prev=None, seq_len=1):
        seq_len = np.int32(seq_len)
        if state_prev is None:
            state_prev = self.sess.run(self.initial_state)
        cwm_vae_feed = {self.input_x: x,
                        self.input_a_prev: a_prev,
                        self.initial_state: state_prev,
                        self.seq_length: seq_len}
        (z_map_s, final_state_s) = self.sess.run([self.z_map_s, self.final_state_s], feed_dict=cwm_vae_feed)
        return z_map_s, final_state_s

    def encode_new_r(self, x, a_prev, r_prev, r_next, state_prev=None, seq_len=1):
        seq_len = np.int32(seq_len)
        if state_prev is None:
            state_prev = self.sess.run(self.initial_state)
        cwm_vae_feed = {self.input_x: x,
                        self.input_a_prev: a_prev,
                        self.input_r_prev: r_prev,
                        self.input_r_next: r_next,
                        self.initial_state: state_prev,
                        self.seq_length: seq_len}
        (z_map_r, final_state_r) = self.sess.run([self.z_map_r, self.final_state_r], feed_dict=cwm_vae_feed)
        return z_map_r, final_state_r

    def encode_mu_s_logvar(self, x, a_prev, seq_len=1):
        seq_len = np.int32(seq_len)
        cwm_vae_feed = {self.input_x: x,
                        self.input_a_prev: a_prev,
                        self.seq_length: seq_len}
        (logmix_s, mu_s, logstd_s) = self.sess.run([self.out_logmix_s, self.out_mean_s, self.out_logstd_s],
                                                feed_dict=cwm_vae_feed)
        return logmix_s, mu_s, logstd_s
    
    def encode_mu_r_logvar(self, x, a_prev, r_prev, r_next, seq_len=1):
        seq_len = np.int32(seq_len)
        cwm_vae_feed = {self.input_x: x,
                        self.input_a_prev: a_prev,
                        self.input_r_prev: r_prev,
                        self.input_r_next: r_next,
                        self.seq_length: seq_len}
        (logmix_r, mu_r, logstd_r) = self.sess.run([self.out_logmix_r, self.out_mean_r, self.out_logstd_r],
                                                feed_dict=cwm_vae_feed)
        return logmix_r, mu_r, logstd_r

    def decode(self, z_s, seq_len=1):
        return self.sess.run(self.y_o, feed_dict={self.z_s: z_s, self.seq_length: seq_len})

    def decode_new(self, z_map_s, seq_len=1):
        return self.sess.run(self.y_o, feed_dict={self.z_s: z_map_s, self.seq_length: seq_len})

    # predict next observation
    def predict(self, z_s, seq_len=1):
        seq_len = np.int32(seq_len)
        return self.sess.run(self.y_o_next, feed_dict={self.z_s: z_s, self.seq_length: seq_len})

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p * 10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def tf_lognormal(self, y, mean, logstd):
        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

    def get_lossfunc(self, logmix, mean, logstd, y):
        v = logmix + self.tf_lognormal(y, mean, logstd)
        v = tf.reduce_logsumexp(v, 1, keepdims=True)
        return -tf.reduce_mean(v)

    def get_mdn_coef(self, output):
        logmix, mean, logstd = tf.split(output, 3, 1)
        logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
        return logmix, mean, logstd

    def kl_gmm(self, logmix, mu, logvar):
        kl_loss = - 0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
        kl_loss = tf.maximum(kl_loss, self.hps.kl_tolerance * self.hps.z_size)
        kl_loss = tf.multiply(tf.exp(logmix), tf.exp(-kl_loss))
        return kl_loss

    def weight_normalization(self):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                p = self.sess.run(var)
                weightnorm_op, pl = self.weightnorm_ops[var]
                self.sess.run(weightnorm_op, feed_dict={pl.name: p})

    def deconv_weightnorm(self, weight):
        if len(weight.get_shape().as_list()) == 2:
            weight = tf.nn.l2_normalize(weight, [0])
        elif len(weight.get_shape().as_list()) == 4:
            weight = tf.nn.l2_normalize(weight, [0, 1, 3])
        return weight

    def conv_weightnorm(self, weight):
        if len(weight.get_shape().as_list()) == 2:
            weight = tf.nn.l2_normalize(weight, [0])
        elif len(weight.get_shape().as_list()) == 4:
            weight = tf.nn.l2_normalize(weight, [0, 1, 2])
        return weight

    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up
        return rparam

    def set_model_params(self, params, is_dyn=False, is_testing=False):
        if is_testing:
            with self.g.as_default():
                t_vars = tf.trainable_variables()
                idx = 0
                for var in t_vars:
                    if var.name.startswith('SSL/A') or \
                            var.name.startswith('SSL/B') or \
                            var.name.startswith('SSL/C') or \
                            var.name.startswith('SSL/D1') or \
                            var.name.startswith('SSL/D2') or \
                            var.name.startswith('SSL/E') or \
                            var.name.startswith('SSL/F') or \
                            var.name.startswith('SSL/G'):
                        pshape = tuple(var.get_shape().as_list())
                        p = np.array(params[idx])
                        assert pshape == p.shape, "inconsistent shape"
                        assign_op, pl = self.assign_ops[var]
                        self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                    idx += 1
        else:
            with self.g.as_default():
                t_vars = tf.trainable_variables()
                idx = 0
                
                for var in t_vars:
                    pshape = tuple(var.get_shape().as_list())
                    p = np.array(params[idx])
                    assert pshape == p.shape, "inconsistent shape"
                    assign_op, pl = self.assign_ops[var]
                    self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                    idx += 1

    def load_json(self, jsonfile='vae.json', is_dyn=False, is_tesing=False):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params, is_dyn, is_tesing)

    def save_json(self, jsonfile='vae.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

    def save_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, 'vae')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, 0)  # just keep one