import tensorflow as tf
import numpy as np
from gym.spaces import Box

from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy
from stable_baselines.common.tf_layers import mlp

import json

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def gaussian_likelihood(input_, mu_, log_std):
    """
    Helper to computer log likelihood of a gaussian.
    Here we assume this is a Diagonal Gaussian.

    :param input_: (tf.Tensor)
    :param mu_: (tf.Tensor)
    :param log_std: (tf.Tensor)
    :return: (tf.Tensor)
    """
    pre_sum = -0.5 * (((input_ - mu_) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def gaussian_entropy(log_std):
    """
    Compute the entropy for a diagonal Gaussian distribution.

    :param log_std: (tf.Tensor) Log of the standard deviation
    :return: (tf.Tensor)
    """
    return tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = tf.cast(input_ > upper, tf.float32)
    clip_low = tf.cast(input_ < lower, tf.float32)
    return input_ + tf.stop_gradient((upper - input_) * clip_up + (lower - input_) * clip_low)


def apply_squashing_func(mu_, pi_, logp_pi):
    """
    Squash the output of the Gaussian distribution
    and account for that in the log probability
    The squashed mean is also returned for using
    deterministic actions.

    :param mu_: (tf.Tensor) Mean of the gaussian
    :param pi_: (tf.Tensor) Output of the policy before squashing
    :param logp_pi: (tf.Tensor) Log probability before squashing
    :return: ([tf.Tensor])
    """
    # Squash the output
    deterministic_policy = tf.tanh(mu_)
    policy = tf.tanh(pi_)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS), axis=1)
    return deterministic_policy, policy, logp_pi


class CNNModel():
    def __init__(self, config, target_space_shape, pretrained_model):
        self.config = config

        self.target_space_shape = target_space_shape

        self.pretrained_model = pretrained_model

    def conv(self, input_tensor, scope, *, n_filters, filter_size, stride,
             pad='SAME', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
        """
        Creates a 2D convolutional layer for TensorFlow

        :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
        :param scope: (str) The TensorFlow variable scope
        :param n_filters: (int) The number of filters
        :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
        or the height and width of kernel filter if the input is a list or tuple
        :param stride: (int) The stride of the convolution
        :param pad: (str) The padding type ('VALID' or 'SAME')
        :param init_scale: (int) The initialization scale
        :param data_format: (str) The data format for the convolution weights
        :param one_dim_bias: (bool) If the bias should be one dimentional or not
        :return: (TensorFlow Tensor) 2d convolutional layer
        """
        if isinstance(filter_size, list) or isinstance(filter_size, tuple):
            assert len(filter_size) == 2, \
                "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
            filter_height = filter_size[0]
            filter_width = filter_size[1]
        else:
            filter_height = filter_size
            filter_width = filter_size
        if data_format == 'NHWC':
            channel_ax = 3
            strides = [1, stride, stride, 1]
            bshape = [1, 1, 1, n_filters]
        elif data_format == 'NCHW':
            channel_ax = 1
            strides = [1, 1, stride, stride]
            bshape = [1, n_filters, 1, 1]
        else:
            raise NotImplementedError


        bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
        n_input = input_tensor.get_shape()[channel_ax].value
        wshape = [filter_height, filter_width, n_input, n_filters]
        with tf.variable_scope(scope):
            weight = tf.get_variable("w", wshape, initializer=self.ortho_init(init_scale))
            bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
            if not one_dim_bias and data_format == 'NHWC':
                bias = tf.reshape(bias, bshape)
            return bias + tf.nn.conv2d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)


    def get_trainable_vars(self, name):
        """
        returns the trainable variables

        :param name: (str) the scope
        :return: ([TensorFlow Variable])
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def conv_to_fc(self, input_tensor):
        """
        Reshapes a Tensor from a convolutional network to a Tensor for a fully connected network

        :param input_tensor: (TensorFlow Tensor) The convolutional input tensor
        :return: (TensorFlow Tensor) The fully connected output tensor
        """
        n_hidden = np.prod([v.value for v in input_tensor.get_shape()[1:]])
        input_tensor = tf.reshape(input_tensor, [-1, n_hidden])
        return input_tensor


    def ortho_init(self, scale=1.0):
        """
        Orthogonal initialization for the policy weights

        :param scale: (float) Scaling factor for the weights.
        :return: (function) an initialization function for the weights
        """

        # _ortho_init(shape, dtype, partition_info=None)
        def _ortho_init(shape, *_, **_kwargs):
            """Intialize weights as Orthogonal matrix.

            Orthogonal matrix initialization [1]_. For n-dimensional shapes where
            n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
            corresponds to the fan-in, so this makes the initialization usable for
            both dense and convolutional layers.

            References
            ----------
            .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
                   "Exact solutions to the nonlinear dynamics of learning in deep
                   linear
            """
            # lasagne ortho init for tf
            shape = tuple(shape)
            if len(shape) == 2:
                flat_shape = shape
            elif len(shape) == 4:  # assumes NHWC
                flat_shape = (np.prod(shape[:-1]), shape[-1])
            else:
                raise NotImplementedError
            gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
            weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
            weights = weights.reshape(shape)
            return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)

        return _ortho_init



    def linear(self, input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0):
        """
        Creates a fully connected layer for TensorFlow

        :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer
        :param scope: (str) The TensorFlow variable scope
        :param n_hidden: (int) The number of hidden neurons
        :param init_scale: (int) The initialization scale
        :param init_bias: (int) The initialization offset bias
        :return: (TensorFlow Tensor) fully connected layer
        """
        with tf.variable_scope(scope):
            n_input = input_tensor.get_shape()[1].value
            weight = tf.get_variable("w", [n_input, n_hidden], initializer=self.ortho_init(init_scale))
            bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(init_bias))
            return tf.matmul(input_tensor, weight) + bias


    def cnn_architecture(self, scaled_images, **kwargs):

        with tf.variable_scope("cnn_model", reuse=tf.AUTO_REUSE): 

            activ = tf.nn.relu
            scaled_images = tf.contrib.layers.batch_norm(scaled_images)

            self.layer_1 = activ(self.conv(scaled_images, 'c1', n_filters=self.config['cnn_params']['l1']['n_filters'], 
                filter_size=self.config['cnn_params']['l1']['filter_size'], stride=self.config['cnn_params']['l1']['stride'], 
                init_scale=self.config['cnn_params']['l1']['init_scale'], **kwargs))

            self.layer_2 = tf.nn.max_pool(value=self.layer_1, ksize=2, strides=[1,2,2,1], padding='VALID', data_format='NHWC')

            # self.layer_2 = tf.contrib.layers.batch_norm(self.layer_2)

            self.layer_3 = activ(self.conv(self.layer_2, 'c2', n_filters=self.config['cnn_params']['l2']['n_filters'], 
                filter_size=self.config['cnn_params']['l2']['filter_size'], stride=self.config['cnn_params']['l2']['stride'], 
                init_scale=self.config['cnn_params']['l2']['init_scale'], **kwargs))

            self.layer_4 = tf.nn.max_pool(value=self.layer_3, ksize=2, strides=[1,2,2,1], padding='VALID', data_format='NHWC')

            # self.layer_4 = tf.contrib.layers.batch_norm(self.layer_4)

            self.layer_5 = activ(self.conv(self.layer_4, 'c3', n_filters=self.config['cnn_params']['l3']['n_filters'], 
                filter_size=self.config['cnn_params']['l3']['filter_size'], stride=self.config['cnn_params']['l3']['stride'], 
                init_scale=self.config['cnn_params']['l3']['init_scale'], **kwargs))

            self.layer_6 = tf.nn.max_pool(value=self.layer_5, ksize=2, strides=[1,2,2,1], padding='VALID', data_format='NHWC')

            # self.layer_6 = tf.contrib.layers.batch_norm(self.layer_6)

            self.layer_7 = self.conv_to_fc(self.layer_6)

            if self.pretrained_model is not None: 
                self.layer_8 = tf.stop_gradient(activ(self.linear(self.layer_7, 'fc1', n_hidden=self.config['cnn_params']['fc']['n_hidden'], init_scale=self.config['cnn_params']['fc']['init_scale'])))
            else: 
                self.layer_8 = activ(self.linear(self.layer_7, 'fc1', n_hidden=self.config['cnn_params']['fc']['n_hidden'], init_scale=self.config['cnn_params']['fc']['init_scale']))

            # self.layer_8 = activ(self.linear(self.layer_7, 'fc1', n_hidden=self.config['cnn_params']['fc']['n_hidden'], init_scale=self.config['cnn_params']['fc']['init_scale']))

            self.layer_9 = tf.layers.dense(self.layer_8, units=self.target_space_shape, activation=None)

            # TODO FIX
            return self.layer_8, self.layer_9



class SACPolicy(BasePolicy):
    """
    Policy object that implements a SAC-like actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, scale=False):
        super(SACPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale)
        assert isinstance(ac_space, Box), "Error: the action space must be of type gym.spaces.Box"

        self.qf1 = None
        self.qf2 = None
        self.value_fn = None
        self.policy = None
        self.deterministic_policy = None
        self.act_mu = None
        self.std = None

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """
        Creates an actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: (TensorFlow Tensor) the output tensor
        """
        raise NotImplementedError

    def make_critics(self, obs=None, action=None, reuse=False,
                     scope="values_fn", create_vf=True, create_qf=True):
        """
        Creates the two Q-Values approximator along with the Value function

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param action: (TensorFlow Tensor) The action placeholder
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :param create_vf: (bool) Whether to create Value fn or not
        :param create_qf: (bool) Whether to create Q-Values fn or not
        :return: ([tf.Tensor]) Mean, action and log probability
        """
        raise NotImplementedError

    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float]) actions
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability params (mean, std) for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float])
        """
        raise NotImplementedError


class FeedForwardPolicy(SACPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param reg_weight: (float) Regularization loss weight for the policy parameters
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", reg_weight=0.0,
                 layer_norm=False, act_fun=tf.nn.relu, config=None, pretrained_model=None, target_shape=6, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse, scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        self.layers = layers
        if layers is None:
            layers = [512, 512]
        self.reg_loss = None
        self.reg_weight = reg_weight
        self.entropy = None

        self.model = CNNModel(config, target_shape, pretrained_model)
        self.cnn_extractor = self.model.cnn_architecture

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h, _ = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)

            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            # self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None, use_bias=False)

            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            # log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None, use_bias=False)
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)


        # Regularize policy output (not used for now)
        # reg_loss = self.reg_weight * 0.5 * tf.reduce_mean(log_std ** 2)
        # reg_loss += self.reg_weight * 0.5 * tf.reduce_mean(mu ** 2)
        # self.reg_loss = reg_loss

        # OpenAI Variation to cap the standard deviation
        # activation = tf.tanh # for log_std
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        # Original Implementation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        # MISSING: reg params for log and mu
        # Apply squashing and account for it in the probability
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                critics_h, _ = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                critics_h = tf.layers.flatten(obs)

            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp(critics_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    value_fn = tf.layers.dense(vf_h, 1, name="vf")
                self.value_fn = value_fn

            if create_qf:
                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_h, action], axis=-1)

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class LnCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(LnCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="cnn", layer_norm=True, **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class LnMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(LnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="mlp", layer_norm=True, **_kwargs)


register_policy("CnnPolicy", CnnPolicy)
register_policy("LnCnnPolicy", LnCnnPolicy)
register_policy("MlpPolicy", MlpPolicy)
register_policy("LnMlpPolicy", LnMlpPolicy)
