"""
This module contains a set of general utility functions for working with the RBM 
model.

Author: @wingr
Date: 2018-02-23
"""
import argparse


def get_argparser():
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
    parser.add_argument('--visualize', type=bool, default=True,
                        help="""Whether or not to show prediction distributions""")
    parser.add_argument('--save_weights', type=bool, default=False,
                        help="""Whether or not to save weights""")
    return parser
