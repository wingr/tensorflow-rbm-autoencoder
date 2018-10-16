"""
This module provides a set of functions to help visualize the training process
and results when working with the RBM model.

Author: @wingr
Date:   2017-10-16
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def all_digits_sample(X, y_integer, height, width):
    """
    This function plots 10 sample images, one for each digit. This is used
    as a quick check that the data loaded correctly.
    Args:
        :param X:           The data array with the sample arrays 
        :type X:            numpy array
        :param y_integer:   The data vector with the integer labels for each 
                            sample. The integers are [0, 9]
        :type y_integer:    numpy 1D array
        :param height:      The image height
        :type height:       Integer
        :param width:       The image width
        :type width:        Integer
    Returns:
        :param fig:         The resulting figure with 10 sample images
        :type fig:          matplotlib.pyplot figure
    """
    fig, _ = plt.subplots(2, 5, figsize=(15, 6))
    digit_list = np.unique(y_integer).astype(int)
    salt_max = 1000
    for digit, ax in enumerate(fig.axes):
        idx = np.where(y_integer == digit)[0][0]
        salt = np.random.randint(0, salt_max)
        image_array = X[idx + salt].reshape(height, width)

        ax.imshow(image_array, cmap='binary')
        ax.set_title('This is a {}'.format(y_integer[idx]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    return fig


def single_digit_sample(X, y_integer, height, width, n_rows=2, n_cols=5):
    """
    This function plots 10 sample images all for the same digit. This is
    used as a quick check that the data loaded correctly.
    Args:
        :param X:           The data array with the sample arrays (1 digit only)
        :type X:            numpy array
        :param y_integer:   The data vector with the integer label for each 
        :type y_integer:    numpy 1D array
        :param height:      The image height
        :type height:       Integer
        :param width:       The image width
        :type width:        Integer
    Returns:
        :param fig:         The resulting figure with 10 sample images
        :type fig:          matplotlib.pyplot figure
    """
    fig, _ = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    array_length = y_integer.shape[0]
    for idx, ax in enumerate(fig.axes):
        idx = np.random.randint(0, array_length)
        image_array = X[idx].reshape(height, width)

        ax.imshow(image_array, cmap='binary')
        ax.set_title('This is a {}'.format(y_integer[idx]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    return fig


def save_image(X_i, y_i, height, width, multiplier, image_dir):
    """
    This function takes our data and saves it as a pictoral representation.
    By default the image is 28 pixels wide by 28 pixels high and is a black
    and white image. The image will be saved in the image_dir with a name 
    following the format: image_label.png where _label is the digit the
    image represents.
    Args:
        :param X_i:         The data array to be saved as an image its shape 
                            is 1 x 784 by default (1 x height*width).
        :type X_i:          Numpy array
        :param y_i:         The target (class) label for the sample as an 
                            integer [0, 9]
        :type y_i:          Numpy 1D array
        :param height:      The height of the image
        :type height:       Integer
        :param width:       The width of the image
        :type width:        Integer
        :param multiplier:  The multiplier to enlarge the image
        :type multiplier:   Integer
        :param image_dir:   The directory into which the image will be saved
        :type image_dir:    String
    Returns: 
        Nothing
    """
    # Create image from data sample
    GRAYSCALE_MAX = 255 # X_i is only 0 or 1s and we need [0, 255] for image
    image_array = GRAYSCALE_MAX * X_i.reshape(height, width)
    image = Image.fromarray(image_array.astype('uint8'))

    # Convert to 'RGB' to get grayscale (not black/white) for invert() to work
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Invert grayscale so black show activation on white background
    image = ImageOps.invert(image)

    # Resize to make it easier to see
    new_h = height * multiplier
    new_w = width * multiplier
    image = image.resize((new_w, new_h)) # columns first not rows for PIL

    # Save image
    if not os.path.exists(image_dir): 
        os.makedirs(image_dir)
    filename = os.path.join(image_dir, 'image_{}.png'.format(int(y_i)))
    image.save(filename)


def plot_training_errs(errs, errs_valid):
    """
    This function plots the training progress using the errors at each iteration
    between the training data and the reconstruction of that data by the RBM
    model.
    Args:
        :param errs:    The array of errors for each iteration
        :type errs:     Float array
    Return:
        :param fig:     The resulting matplotlib figure for trainining 
        :type fig:      Matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(errs, 'b-', label='Training')
    ax.plot(errs_valid, 'r-', label='Validation')
    ax.set_title('Training Progress')
    ax.set_xlabel('Training iterations')
    ax.set_ylabel('Error')
    ax.legend(loc=0, labelspacing=1.5, fontsize=12)
    ax.grid(True)
    return fig


def scale_to_unit_interval(ndar, eps=1e-8):
    """
    Scales all values in the ndarray ndar to be between 0 and 1
    Args:
        :param ndar:    The original ndarray
        :type ndar:     Float array
        :param eps:     A small constant to help with divide by zero errors
    Returns:
        :param ndar:    A scaled version of ndar
        :type ndar:     Float array
    """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).
    Args:
        :param X:           A 2-D array in which every row is a flattened image.
        :type X:            A 2-D ndarray or a tuple of 4 channels, elements of 
                            which can be 2-D ndarrays or None
        :param img_shape:   The original shape of each image
        :type img_shape:    Tuple (height, width)
        :param tile_shape:  The number of images to tile (rows, cols)
        :type tile_shape:   Tuple (rows, cols)
        :param output_pixel_vals:           If output should be pixel values 
                                            (i.e. int8 values) or floats
        :param scale_rows_to_unit_interval: If the values need to be scaled 
                                            before being plotted to [0,1] or not
    Returns: 
        :param out_array:   Array suitable for viewing as an image.
                            (See:`Image.fromarray`.)
        :type out_array:    A 2-d array with same dtype as X.

    This function was heavily adapted from,
    https://github.com/Cospel/rbm-ae-tf/blob/master/util.py
    Retrieved on 2017-10-17
    """
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [0, 0]
    out_shape[0] = (img_shape[0] + tile_spacing[0]) \
                    * tile_shape[0] - tile_spacing[0]
    out_shape[1] = (img_shape[1] + tile_spacing[1]) \
                    * tile_shape[1] - tile_spacing[1]

    # if we are dealing with only one channel
    H, W = img_shape
    Hs, Ws = tile_spacing

    # generate a matrix to store the output
    dt = X.dtype
    if output_pixel_vals:
        dt = 'uint8'
    out_array = np.zeros(out_shape, dtype=dt)

    for tile_row in range(tile_shape[0]):
        for tile_col in range(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                this_x = X[tile_row * tile_shape[1] + tile_col]
                if scale_rows_to_unit_interval:
                    # if we should scale values to be between 0 and 1
                    # do this by calling the `scale_to_unit_interval`
                    # function
                    this_img = scale_to_unit_interval(
                        this_x.reshape(img_shape))
                else:
                    this_img = this_x.reshape(img_shape)
                # add the slice to the corresponding position in the
                # output array
                c = 1
                if output_pixel_vals:
                    c = 255
                out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                ] = this_img * c
    return out_array


def get_weight_tiles(W, img_shape, tile_shape):
    """
    This function creates a tile plot for the weights on the hidden units in
    the RBM. If there are, e.g. 400 hidden units and each training image is
    28 pixels x 28 pixels, then the resulting figure will be 20 x 20 tiles,
    each with an image 28 x 28 pixels big.
    Args:
        :param W:           The array of weights for the hidden units
        :type W:            Float array
        :param image_shape: The shape tuple for the image, e.g. 28 x 28
        :type image_shape:  Tuple
        :param tile_shape:  The shape for the resulting set of tiles, 20 x 20
        :type tile_shape:   Tuple
    Returns:
        :param fig:         The resulting tile plot
        :type fig:          A matplotlib figure object


    NOTE: For some reason this doesn't work with matplotlib=2.1.2. I had to 
    downgrade to 2.0.0. The error is,
    `AttributeError: 'numpy.ndarray' object has no attribute 'mask'`
    See also, https://stackoverflow.com/questions/47992078/error-trying-to-split-image-colors-numpy-ndarray-object-has-no-attribute-mask
    """
    image = Image.fromarray(tile_raster_images(X=W.T, img_shape=img_shape, 
        tile_shape=tile_shape, tile_spacing=(1, 1)))
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return fig


def round_to_nearest_n(x, base=10):
    """
    This is a helper function that will round a value (x) to the nearest number
    specified by base. For example, round_to_nearest_n(36, base=10) -> 40
    whereas round_to_nearest_n(36, base=5) -> 35.
    ARGS:
        x       = The number to round
        base    = What number to round to, e.g. round to nearest 10
    RETURN:
        The rounded number
    """
    return int(base * round(float(x)/base))


def plot_hist(x, feature, sub_lbl='', n_round = 5.0):
    """
    This function plots a histogram of x.
    Args:
        :param x:       The array of data to plot
        :type x:        Float array
        :param feature: The label to use for the x-label and title of the plot
        :type feature:  String
        :param sub_lbl: The subtitle for the plot
        :type sub_lbl:  String
        :param n_round: The value to use to round data. For example, if 
                        n_round = 5.0, then all data will be rounded to the
                        nearest value of 5, 23 --> 25, 92 --> 90
        :type n_round:  Float
    Returns:
        :param fig:     The resulting figure
        :type fig:      A matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.hist(x, bins=mybins, align='left') #left centers it on label
    ax.set_xticks(mybins)
    ax.set_xlim([-0.5, max_val])
    ax.set_title('Feature distribution (%s)\n%s' % (feature, sub_lbl), fontsize=16)
    ax.set_xlabel(feature, fontsize=14)
    ax.set_ylabel('Org count', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which='both')
    return fig


def energy_distributions(neg_energies, y_integer, digit, n_rows=2, n_cols=5):
    """
    This function plots the negative free engergies (basically the
    non-normalized probabilities) for each sample in our training set in
    relation to the trained RBM. For example, if we train the RBM on the digit
    0, then we look at each sample and basically ask, "what is the probability
    that this fits our RBM model, i.e. that this digit is a 0?" The negative
    engergies are scaled between [0, 1.0] and a higher number represents a
    higher probability that the sample matches the trained RBM. In this case,
    we expect the actual 0s to have higher negative engergies than the other
    digits.
    Args:
        :param neg_engergies:   The array of negative engergies for each sample
                                in relation to the trained RBM.
        :type neg_engergies:    Float array
        :param y_integer:       The array of target values [0, 9]
        :type y_integer:        Int array
        :param digit:           The digit used to train the RBM [0, 9]
        :type digit:            Int
        :param n_rows:          The number of rows for the plot
        :type n_rows:           Int
        :param n_cols:          The number of columns for the plot
        :type n_cols:           Int
    """
    mybins = np.linspace(0.0, 1.0, 50)
    neg_energies /= neg_energies.max()
    fig, _ = plt.subplots(n_rows, n_cols, figsize=(18, 12), 
                sharex=False, sharey=True)
    title = "Scaled Negative Free Energies - trained on {}'s" \
            "\n(higher is better)".format(digit)
    fig.subplots_adjust(hspace=0.3)
    fig.suptitle(title, fontsize=14)

    for idx, ax in enumerate(fig.axes):
        x = neg_energies[y_integer == idx]
        x_mean = x.mean()
        t = "Distribution for {}\n(mean = {:.3f})".format(idx, x_mean)
        ax.hist(x, bins=mybins, align='left') #left centers it on label
        ax.axvline(x=x_mean, lw=2.0, color='red', ls='--')
        ax.set_title(t)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Scaled negative free energy")
        if idx % n_cols == 0:
            ax.set_ylabel("Number of samples")
    return fig
