"""
This module implements a Restricted Boltzmann Machine. This class is meant to
be imported and used by another module, e.g. the `run_rbm.py` module.

Author: @wingr (see note below)
Date: 2018-02-23

Note: This was heavily adapted from
  - https://github.com/meownoid/tensorfow-rbm
  - https://github.com/Cospel/rbm-ae-tf
"""
import sys

import numpy as np
import tensorflow as tf


class RBM:
    def __init__(self, 
                n_visible, 
                n_hidden, 
                learning_rate=0.01,
                momentum=0.95):
        """
        This function intializes the class object. It assigns the values passed
        in the function call and sets up the tensorflow variables and placeholders
        that will be used during training. It also initializes a set of operations
        that are defined using the _initialize_vars() function. These operations
        provide the functionality needed during training. Lastly it initializes
        everything and creates a tensorflow session.
        Args:
            :param n_visible:       The number of visible units - for images this
                                    corresponds to the number of pixels
            :type n_visiable:       Integer
            :param n_hidden:        The number of hidden units
            :type n_hidden:         Integer
            :param learning_rate:   The learning rate used during training
            :type learning_rate:    Float
            :param momentum:        The momentum used during training - momentum 
                                    helps the gradient descent work more quickly
                                    and helps avoid oscillating behavior.
            :type momentum:         Float
        Returns:
        """
        # Define the placeholders and variables that will be used
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.X = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])
        self.v_sample = tf.placeholder(tf.float32, [None, self.n_visible])

        self.W = tf.get_variable('W', shape=[self.n_visible, self.n_hidden],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), 
                    dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), 
                    dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), 
                    dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), 
                    dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), 
                    dtype=tf.float32)

        # Create variables for the operators that will be used during training
        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None

        # Define the operators in this function
        self._initialize_vars()

        # Ensure that all operators have been defined
        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None
       
        # Define a couple other functions to use with the trained model
        self.compute_err = tf.reduce_mean(tf.square(self.X - self.compute_visible))
        self.compute_free_energy = self._free_energy(self.v_sample)

        # Initialize variables and create a session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _sample_bernoulli(self, probs):
        """
        This function provides the sampling step for the Gibbs sampling used
        in the recreation of the visible layer from the hidden layer. 
        Args:
            :param probs:   The probabilities for the hidden layer computed
                            from the visible layer values (the data), the
                            weight matrix (being learned during training), and
                            the bias on the hidden units (also being learned)
            :type probs:    float matrix
        Returns:
            An array of the reconstructed visible binary values
        A couple notes:
            - The probabilities are floats and must be converted to binary values
            - This conversion is accomplished by first asking whether the 
              probability value is above a random number uniformly distributed
              between 0 and 1 (see section 3.1 of Hinton's Practical Guide to
              Training Restricted Boltzman Machines, 
              https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf). This is 
              the subtraction part.
            - Next we use tf.sign to convert the resulting floats to values that
              are either -1 or 1.
            - Lastly, we use the tf.relu function to convert the -1 values to 0
              Thereby only giving values of 0 or 1 for the result.
        """
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))


    def _update(self, delta_old, model_err):
        """
        This function provides the update rule (gradient) using the learning 
        rate and momentum. This function is used to update the weights and 
        biases during training. The total gradient is divided by the size of 
        the mini-batch following the advice given in section 4 of Hinton's 
        Practical Guide to Training Restricted  Boltzman Machines, 
        https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf. 
        For more information about momentum, see section 9 of the same guide.
        Args:
            :param delta_old:   The old value of the weights or biases
            :type delta_old:    float matrix / vector
            :param model_err:   The error is calculated from the difference in
                                the expected value considering the visible 
                                layer (data) and the probabilities on the hidden
                                layer, this gives the positive step, and the
                                expected value considering the reconstructed
                                visible layer and the reconstructed hidden layer
                                probabilities (the negative step). See section
                                2 of the guide linked above.
            :type model_err:    float array
        Returns:
            :param delta:       The new value for the weights or biases
            :type delta:        float matrix / vector
        """
        delta = self.momentum * delta_old \
                + self.learning_rate / tf.to_float(tf.shape(model_err)[0]) \
                * model_err
        return delta


    def _free_energy(self, v_sample):
        """
        Function to compute the free energy. Conceptually, the free energy can 
        be used like a non-normalized probability value, or log-likelihood, for 
        a given sample relative to the trained RBM. The lower the free engergy 
        the higher the likelihood the sample fits the training data.
        The free energy computation can be seen in this tutorial, 
        http://deeplearning.net/tutorial/rbm.html#contrastive-divergence-cd-k
        and the implementation on which this function was based can be seen here,
        https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/rbm.py#L130
        Args:
            :param v_sample:    A sample binary vector representing the visible
                                layer (sample of data)
            :type v_sample:     binary vector
        Returns:
            The free energy of the sample vector relative to the trained RBM
        """
        wx_b = tf.matmul(v_sample, self.W) + self.hidden_bias
        vbias_term = tf.multiply(v_sample, self.visible_bias)
        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(wx_b)), axis=1)
        return tf.transpose(-tf.transpose(vbias_term) - hidden_term)


    def _initialize_vars(self):
        """
        This function initializes the placeholders, variables, and operators
        from the __init__() function. This function contains most of the logic
        for training the RBM.

        Notes:
        The tricky part of this function is the part that deals with the
        contrastive divergence step to update the weights. The contrastive
        divergence algorithm uses the difference between the expectations in
        the distributions produced by the data (the positive_grad below) and
        the expectations in the distributions produced by the model, or really
        a single reconstruction step using Gibbs sampling (the negative_grad).
        The positive_grad = <v_i * h_j>_{data}, visible vector * eq. 7 result
        and negative_grad = <v_i * h_j>_{recon}, visible_recon * hidden_recon.
        See A Practical Guide to Training Restricted  Boltzman Machines, 
        section 2, https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf for 
        more information on the details.
        """
        hidden_p = tf.nn.sigmoid(tf.matmul(self.X, self.W) + self.hidden_bias)
        visible_recon_p = tf.nn.sigmoid(tf.matmul(
            self._sample_bernoulli(hidden_p), 
            tf.transpose(self.W)) + self.visible_bias)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.W) \
            + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(self.X), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        delta_w_new = self._update(self.delta_w, (positive_grad - negative_grad))
        delta_visible_bias_new = self._update(self.delta_visible_bias, 
            tf.reduce_mean(self.X - visible_recon_p, 0))
        delta_hidden_bias_new = self._update(self.delta_hidden_bias, 
            tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias\
            .assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias\
            .assign(delta_hidden_bias_new)

        update_w = self.W.assign(self.W + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias \
            + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias \
            + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, 
            update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, 
            update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.X, self.W) \
            + self.hidden_bias)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, 
            tf.transpose(self.W)) + self.visible_bias)
        self.compute_visible_from_hidden = tf.matmul(self.y, 
            tf.transpose(self.W)) + self.visible_bias


    def _partial_fit(self, batch_x):
        """
        This function runs a training iteration on the mini-batch sample from
        the training data.
        Args:
            :param batch_x: The mini-batch from our training data
            :type batch_x:  An array of binary vectors the size of our mini-batch
        """
        self.sess.run(self.update_weights + self.update_deltas, 
            feed_dict={self.X: batch_x})


    def fit(self, data_x, x_valid, n_epochs=10, batch_size=100, verbose=True):
        """
        This function fits the RBM and prints progress to the screen if
        verbose=True.
        Args:
            :param data_x:      The full array of our training data, binary
                                vectors
            :type data_x:       Matrix of binary vectors
            :param n_epochs:    The number of times to iterate through the 
                                training data during training
            :type n_epochs:     Integer
            :param batch_size:  The number of samples to use in each mini-batch
            :type batch_size:   Integer
            :param verbose:     Whether to print training progress to stdout
            :type verbose:      Boolean
        Returns:
            :param errs:        An array of training errors for use in 
                                visualizing training progress.
            :type errs:         Array of floats
        """
        assert n_epochs > 0
        assert batch_size > 0

        n_data = data_x.shape[0]
        n_batches = int(np.ceil(n_data / batch_size))
        errs = []
        errs_valid = []

        for epoch in range(n_epochs):
            if verbose:
                print('Epoch: {}'.format(epoch + 1))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_valid = np.zeros((n_batches,))
            epoch_errs_idx = 0

            # Shuffle data samples
            shuffle_idx = np.random.permutation(data_x.shape[0])
            data_x_shuffled = data_x[shuffle_idx]

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = (batch + 1) * batch_size
                batch_x = data_x_shuffled[start_idx:end_idx]
                self._partial_fit(batch_x)

                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_idx] = batch_err
                validation_err = self.get_err(x_valid)
                epoch_errs_valid[epoch_errs_idx] = validation_err
                epoch_errs_idx += 1

            if verbose:
                err_mean = epoch_errs.mean()
                print('Train error: {:.4f}'.format(err_mean))
                print('Validation error: {:.4f}'.format(epoch_errs_valid.mean()))
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])
            errs_valid = np.hstack([errs_valid, epoch_errs_valid])

        return errs, errs_valid


    def get_err(self, batch_x):
        """
        This function computes the squared error between the data and the 
        reconstruction of the data. This is easy to calculate and gives a
        proxy for the learning progress, but is not a perfect measure of 
        learning. For more information, see section 5 of Hinton's Practical 
        Guide to Training Restricted Boltzman Machines, 
        https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf. 
        Args:
            :param batch_x: The mini-batch of our training data
            :type batch_x:  An array of binary vectors
        Returns:
            The squared error between the data and the reconstruction of that
            data by the trained (or trained at this point) RBM.
        """
        return self.sess.run(self.compute_err, feed_dict={self.X: batch_x})


    def get_weights(self):
        """
        A getter function that returns the weights and biases trained so far.
        Args:
        Returns:
            :param self.W:              The array of current weight values for 
                                        the RBM. This has shape 
                                        self.n_visible x self.n_hidden
            :type self.W:               Float array
            :param self.visible_bias:   The array of biases on the visible units.
                                        This has shape [self.n_visible]
            :type self.visible_bias:    Float array
            :param self.hidden_bias:    The array of biases on the hidden units.
                                        This has shape [self.n_hidden]
            :type self.hidden_bias:     Float array
        """
        return self.sess.run(self.W), \
                self.sess.run(self.visible_bias), \
                self.sess.run(self.hidden_bias)


    def get_free_energy(self, v_sample):
        """
        A getter function that returns the free engergy associated with the 
        passed in v_sample.
        Args:
            :param v_sample:    A binary array representing a sample, e.g. the 
                                vectorized representation of a mnist digit
                                being classified.
            :type v_sample:     Binary array
        Returns:
            The free engergy associated with the sample relative to the trained
            RBM. For example, if the RBM was trainined on the digit 7, then
            this energy would be like a non-normalized probability that the
            v_sample belongs to the same class.
        """
        return self.sess.run(self.compute_free_energy, \
            feed_dict={self.v_sample: v_sample})


    def set_weights(self, W, visible_bias, hidden_bias):
        """
        A setter function to set the weights and biases of the RBM.
        Args:
            :param W:               The array of weight values for the RBM. 
                                    the shape is self.n_visible x self.n_hidden
            :type W:                Float array
            :param visible_bias:    The array of biases on the visible units.
                                    This has shape [self.n_visible]
            :type visible_bias:     Float array
            :param hidden_bias:     The array of biases on the hidden units.
                                    This has shape [self.n_hidden]
            :type hidden_bias:      Float array
        """
        self.sess.run(self.W.assign(W))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))


    def save_weights(self, filename, prefix):
        """
        This function saves the trained weights to a file for reloading.
        Args:
            :param filename:    The file to which to save the weights
            :type filename:     String
            :param prefix:      A prefix for each weight array
            :type prefix        String
        Returns:
        """
        saver = tf.train.Saver({prefix + '_w': self.W,
                                prefix + '_vb': self.visible_bias,
                                prefix + '_hb': self.hidden_bias})
        saver.save(self.sess, filename)


    def load_weights(self, filename, prefix):
        """
        This function loads the trained weights from a file
        Args:
            :param filename:    The file to which to save the weights
            :type filename:     String
            :param prefix:      A prefix for each weight array
            :type prefix        String
        Returns:
        """
        saver = tf.train.Saver({prefix + '_w': self.W,
                                prefix + '_vb': self.visible_bias,
                                prefix + '_hb': self.hidden_bias})
        saver.restore(self.sess, filename)
