"""
This model creates an RBM class instance and trains it. It also contains some
functions for visualizing the training process, the weights on the hidden units
and the free engergies of a set of samples.

Author: @wingr
Date: 2017-10-16

Note: The primary logic is contained in the rbm.py module.
"""

import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import data_utils, visual_utils
from rbm import RBM


def setup_argparser():
    """
    Define an argparser to determine how the RBM will be trained.
    Args:
    Returns:
        :param parser:  The argparse object
        :type parser:   ArgumentParser() instance
    """
    parser = argparse.ArgumentParser(description="""
       This script trains an RBM model on the MNIST digits dataset.""")
    parser.add_argument('--n_epochs', type=int, default=1,
                        help="""Number of training epochs""")
    parser.add_argument('--n_hidden', type=int, default=400,
                        help="""The number of hidden layers used in the RBM""")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="""The number of samples in each mini-batch""")
    parser.add_argument('--digit', type=int, default=0,
                        help="""The MNIST digit on which to train the RBM""")
    return parser


def get_data(data_dir, data_file):
    """
    This function either pulls the MNIST data from sklearn.datasets and stores
    the data in the data_dir. If the data has already been pulled and exists
    in the data_dir, then it is loaded from there instead. The raw data is then
    broken into an X binary array, an y_integer array, and a y_onehot array.
    Args:
        :param data_dir:    The directory into which the MNIST data is saved
                            or from which the existant data is pulled.
        :type data_dir:     String
        :param data_file:   The name of the data file itself. By default when
                            the dataset is pulled from MNIST, it is saved into
                            mldata/mnist-original.mat
        :type data_file:    String
    Returns:
        :param X:           A binary array representation of the hand-written
                            digit 
        :type X:            Integer array
        :param y_integer:   An array of the target variables in the range [0, 9]
        :type y_integer:    Integer array
        :param y_onehot:    A onehot array representation of the target values
        :type y_onehot:     Binary (integer) array
    """
    if os.path.exists(os.path.join(data_dir, data_file)):
        print('Loading existing .mat file')
        X_grayscale, y_integer = data_utils.get_mat_file(data_dir, data_file)
    else:
        print('Pulling new data from mldata...')
        X_grayscale, y_integer = data_utils.get_mldata(data_dir)
    
    X = data_utils.transform_x_to_binary(X_grayscale)
    y_onehot = data_utils.transform_y_to_onehot(y_integer)
    return X, y_integer, y_onehot


def plot_weights(rbm_obj, img_shape, tile_shape, outfile, show_it=False):
    """
    This is a wrapper function to plot the weights for each of the hidden units.
    The plot will show a representation of the digit as learned by each
    of the hidden units.
    Args:
        :param rbm_obj:     An instance of the RBM class
        :type rbm_obj:      RBM object
        :param image_shape: The shape tuple for the image, e.g. 28 x 28
        :type image_shape:  Tuple
        :param tile_shape:  The shape for the resulting set of tiles, 20 x 20
        :type tile_shape:   Tuple
        :param outfile:     The file path to which to save the plot
        :type outfile:      String
    Returns:
    """
    W_final, _, _ = rbm_obj.get_weights()
    fig = visual_utils.get_weight_tiles(W_final, img_shape, tile_shape)
    plt.savefig(outfile)
    if show_it:
        plt.show()


def plot_training_errs(errs, outfile, show_it=False):
    """
    This is a wrapper function to plot the training progress in terms of the
    errors between the training data and the reconstructed visible layer.
    Args:
        :param errs:    The error between the training data and the 
                        reconstructed visible layer
        :type errs:     Float array
        :param outfile: The file path to which to save the plot
        :type outfile:  String
    """
    fig = visual_utils.plot_training_errs(errs)
    plt.savefig(outfile)
    if show_it:
        plt.show()


def plot_digit_engeries(X, rbm_obj, outfile, show_it=False):
    """
    This is a wrapper function to plot the negative free engergies (basically
    the non-normalized probabilities) for each sample in our training set in
    relation to the trained RBM. For example, if we train the RBM on the digit
    0, then we look at each sample and basically ask, "what is the probability
    that this fits our RBM model, i.e. that this digit is a 0?" The negative
    engergies are scaled between [0, 1.0] and a higher number represents a
    higher probability that the sample matches the trained RBM. In this case,
    we expect the actual 0s to have higher negative engergies than the other
    digits.
    Args:
        :param X:       The training data array
        :type X:        Float array
        :param rbm_obj: The trained RBM object
        :type rbm_obj:  RBM object type
        :param outfile: The file path to which to save the plot
        :type outfile:  String
        :param show_it: A boolean for whether the plot should be displayed in
                        addition to being saved
        :type show_it:  Boolean
    Returns:
    """
    v_sample = X
    if v_sample.ndim == 1:
        v_sample = v_sample.reshape(1, len(v_sample))
    sample_p = rbm_obj.get_free_energy(v_sample)
    neg_energies = -1.0 * sample_p.mean(axis=1)
    fig = visual_utils.energy_distributions(neg_energies, y_integer, args.digit)
    plt.savefig(outfile)
    if show_it:
        plt.show()


if __name__ == '__main__':
    start = time.time()

    # ---------------------------- CONTROL PANEL ----------------------------- #
    data_dir = os.path.join(os.getcwd(), 'data/')
    image_dir = os.path.join(os.getcwd(), 'images/')
    weights_dir = os.path.join(os.getcwd(), 'data/saved-weights/')
    data_file = 'mldata/mnist-original.mat'
    height = 28
    width = 28
    save_training = False
    visualize = False

    # --------------------------- SETUP ARG PARSER --------------------------- #
    parser = setup_argparser()
    args = parser.parse_args()

    # ------------------------ OTHER DERIVED VARIABLES ----------------------- #
    img_shape = (height, width)
    tile_side = int(np.sqrt(args.n_hidden))
    tile_shape = (tile_side, tile_side)
    n_visible = height * width
    errs_plot_outfile = os.path.join(image_dir, 
        'training_errors_e{}_h{}_b{}_d{}'\
        .format(args.n_epochs, args.n_hidden, args.batch_size, args.digit))
    weights_plot_outfile = os.path.join(image_dir, 
        'hidden_weights_e{}_h{}_b{}_d{}'\
        .format(args.n_epochs, args.n_hidden, args.batch_size, args.digit))
    energy_plot_outfile = os.path.join(image_dir, 
        'energy_distributions_e{}_h{}_b{}_d{}'\
        .format(args.n_epochs, args.n_hidden, args.batch_size, args.digit))
    weights_file = weights_dir + 'weights_e{}_h{}_b{}_d{}'\
        .format(args.n_epochs, args.n_hidden, args.batch_size, args.digit)

    # -------------------------- GET DATA FOR DIGIT -------------------------- #
    X, y_integer, _ = get_data(data_dir, data_file)
    X_digit = X[y_integer == args.digit]

    # -------------------------- DEFINE AND FIT RBM -------------------------- #
    rbm_obj = RBM(n_visible, args.n_hidden)
    errs = rbm_obj.fit(X_digit, n_epochs=args.n_epochs, batch_size=args.batch_size)
    if save_training:
        rbm_obj.save_weights(weights_file, weights_prefix='weights')

    # ----------------------- ACTIONS WITH TRAINED RBM ----------------------- #
    if visualize:
        print('Creating plots')
        plot_training_errs(errs, errs_plot_outfile)
        plot_weights(rbm_obj, img_shape, tile_shape, weights_plot_outfile)
        plot_digit_engeries(X, rbm_obj, energy_plot_outfile)
    
    delta = (time.time() - start)
    print('Elapsed time = {:.4f} secs ({:.4f} mins)'.format(delta, delta/60.0))
