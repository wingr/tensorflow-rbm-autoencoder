## Overview
This directory contains code to create a simple code pipeline for running Restricted Boltzman Machine (RMB) models. Currently, this runs with the MNIST digits set as an example.

Most of the logic is contained in the `rbm.py` module. The `run_rbm.py` model is the driver module that creates and trains the `RBM()` object instance. The `utils/` directory has utility functions for working with the data and visualizations.

## Running
There is a script, `script/run-digits-rbm`, that runs the RBM model on the MNIST digits data set. This script can be run within the Docker container, which is the intended way for this to be run, or without Docker. Either way, it captures the command line arguments and passes them to the `run_rbm.py` module. This module then creates an instance of the `RBM()` class and trains it according to the specifications passed in.

#### With Docker
The recommended way to run this is through Docker from the top-level directory in this repository.
  * From the top-level directory, run `$ script/app-env script/run-digits-rbm` plus any command line arguments.
  * The `script/app-env` part puts you into the Docker container's environment, and the `script/run-digits-rbm` part is the entry point into the application as mentioned above.

#### Without Docker
The recommended way to run this is using the `script/run-digits-rbm` as the entry point. This can be run without Docker if you have all the required packages (see the top-level README for setup directions).
  * From the top-level directory, run `$ script/run-digits-rbm` plus any command line arguments.

#### Running without the CLI script
The model can also be run without using the CLI script from this directory.
  * `$ python run_rbm.py` plus any command line arguments

#### Command line arguments
Whether the models are run with Docker or without, or with or without the CLI script, there are a number of optional command line arguments for running the models. The command line arguments are as follows:

  * **n_epochs:** The number of epochs to run (number of times through the data). The default is 1.
  * **n_hidden:** The number of hidden nodes to use for the RBM model. The default is 400.
  * **batch_size:** The number of samples to use for each mini-batch. The default is 100.
  * **digit:** A number in [0, 9]. Samples with target values matching this number are used to train the RBM. The default is 0.
  * **visualize:** This boolean argument controls whether the visualizations for the training errors, weights, and free energies are plotted and saved. The default is `True`.
  * **save_weights:** This boolean argument controls whether the trained weights are saved. The default is `False`.

For example (shown with defaults):
```bash
$ python run_rbm.py --n_epochs 1
                    --n_hidden 400
                    --batch_size 100
                    --digit 0
                    --visualize=True
                    --save_weights=False
```
