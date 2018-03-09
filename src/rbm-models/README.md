## Overview
For this project I wanted to create a simple code pipeline for running Restricted Boltzman Machine models.

Most of the logic is contained in the `rbm.py` module. The `run_rbm.py` model is the driver module that creates and trains the RBM() object instance. The `utils/` directory has utility functions for working with the data and visualizations.

## Running
The entry point is the `run_rbm.py` module. This module creates an instance of the RBM() class and trains it according to the specifications passed in. There are a couple hardcoded boolean values that determine if the weights are saved and the images are displayed (in addition to being saved). Both are set to `False` by default.

#### To run:
```bash
$ python run_rbm.py --n_epochs <int>
                     --n_hidden <int>
                     --batch_size <int>
                     --digit <int>
```

For example:
```bash
$ python run_rbm.py --n_epochs 100
                    --n_hidden 400
                    --batch_size 100
                    --digit 7
```

#### Arguments
The command line arguments are as follows:
* **n_epochs:** The number of epochs to run (number of times through the data). The default is 1.
* **n_hidden:** The number of hidden nodes to use for the RBM model. The default is 400.
* **batch_size:** The number of samples to use for each mini-batch. The default is 100.
* **digit:** A number in [0, 9]. Samples with target values matching this number are used to train the RBM. The default is 0. 
