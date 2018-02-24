"""
This module contains a set of utility functions for working with the RBM model.

Author: @wingr
Date: 2017-10-16
"""
import os

import numpy as np
from scipy.io import loadmat
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def get_mldata(data_dir):
    """
    Retrieve the MNIST digits dataset from sklearn.
    Args:
        :param data_dir:        The directory into which to save the data
        :type data_dir:         String
    Returns:
        :param mnist.data:      The pixel values for each data sample
        :type mnist.data:       Float array
        :param mnist.target:    The digit represented by the sample
        :type mnist.target:     Int array
    """
    mnist = datasets.fetch_mldata('MNIST original', data_home=data_dir)
    return [mnist.data, mnist.target.astype(int)]


def get_mat_file(data_dir, data_file):
    """
    This file loads the data pulled by the get_mldata() above if it has been
    previously downloaded. This is much faster than repulling the data.
    Args:
        :param data_dir:    The directory to which the data was saved
        :type data_dir:     String
        :param data_file:   The specific file name of the .mat file
        :type data_dir:     String
    Returns:
        :param X_array:     The array of training samples representing the 
                            pixel values from the image of each digit
        :type X_array:      Int array
        :param y_array:     The array of target values (digit [0, 9]) for 
                            each sample
        :type y_array:      Int array
    """
    mnist_dict = loadmat(os.path.join(data_dir, data_file))
    X_array = np.transpose(mnist_dict['data']) 
    y_array = np.reshape(mnist_dict['label'], mnist_dict['label'].shape[1]).astype(int)
    return [X_array, y_array]
    

def transform_x_to_binary(X_grayscale):
    """
    The pixel values are originally given in a grayscale, integer values 
    between [0, 255], and this function turns those grayscale values into 
    a binary array of only 1s and 0s.
    Args:
        :param X_grayscale: The array of grayscale values
        :type X_grayscale:  Int array
    Returns:
        :param X:           The input array transformed into binary values
        :type X:            Int array
    """
    # Transform grayscale to binary
    X = (X_grayscale > 0.0).astype(int)
    return X


def transform_y_to_onehot(y_integer):
    """
    This function transforms the integer representations of the target values,
    i.e. the digits [0, 9], into one-hot encoded arrays.
    For example, the digit 3 --> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    Args:
        :param y_integer:   An array of digits between [0, 9]
        :type y_integer:    Int array
    Returns:
        :param y:           The one-hot encoded version of each target value
        :type y:            Int array
    """
    label_binarizer = LabelBinarizer()
    y_values = np.unique(y_integer)
    label_binarizer.fit(y_values)
    y = label_binarizer.transform(y_integer)
    return y


def datasets_for_digit(X, y, digit):
    """
    This function selects only samples from the training set X and y that match
    the digit specified.
    Args:
        :param X:       The array of all digit samples
        :type X:        Int array
        :param y:       The matching array of all digit targets
        :type y:        Int array
        :param digit:   The digit to use when downsampling X and y
        :type digit:    Int
    Returns:
        :param X_data:  The subset of X matching digit
        :type X_data:   Int array
        :param y_data:  The subset of y matching digit
        :type y_data:   Int array
    """
    # Select only samples for the given digit
    X_d = X[y == digit]
    y_d = y[y == digit]

    # Split off the test set from the training and validation sets
    X_t_and_v, X_test, y_t_and_v, y_test = train_test_split(
        X_d, y_d, test_size=0.10, shuffle=True, random_state=4)

    # Split the remainder into training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_t_and_v, y_t_and_v, test_size=0.10, shuffle=True, random_state=4)
    x_data = [X_train, X_valid, X_test] 
    y_data = [y_train, y_valid, y_test]
    return [x_data, y_data]

