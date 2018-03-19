## Overview
This repository contains basic Restricted Boltzman Machine (RBM) and autoencoder example models based on Tensorflow and run on the MNIST digits dataset. These examples are designed to run within a Docker container, but can also be run using just a Python virtual environment or similar.

#### Contents

* `.dockerignore`: Files to exclude from the Docker container build
* `.gitignore`: Files to exclude from git tracking
* `Dockerfile`: Build file for the Docker container
* `requirements.txt`: List of Python package dependencies
* `script/`: Directory containing scripts to build the Docker container, run tests, and run the models
* `src/`: Directory containing the source files for the models
  * `src/rbm-models`: Directory containing the source files for the RBM model
    * `src/rbm-models/rbm.py`: This file contains most of the logic for the RBM model itself
    * `src/rbm-models/run_rbm.py`: This is a driver file of sorts that calls the logic from `rbm.py`, trains the model, makes predictions, and creates a few images from the output
    * `src/rbm-models/utils/`: Directory containing some utility functions for use by the RBM model, e.g. functions to retrieve data, plot figures, etc.
* `tests`: Directory containing tests for the models (WIP)

## Setup
This project is intended to be run in Docker, but it can also run with a more standard Python setup.

#### With Docker
To run this project in Docker, you will need Docker installed. To install Docker, follow the install directions [here](https://docs.docker.com/install/).

1. Clone repository
2. To create the Docker container, run `$ script/bootstrap`
3. Test that the container is setup correctly by running , `script/app-env pytest tests/test_docker_setup.py`

#### Without Docker
It is also possible to run these models without Docker if you have the packages from the [requirements.txt](./requirements.txt) installed along with Python 3.6. I would recommend using a [virtual environment](https://virtualenv.pypa.io/en/stable/) if you are not using the Docker container so as not to interfere with your system setup.

1. Create virtual environment (Recommended, but optional)
2. Clone repository
3. Run `$ pip install -r requirements.txt` to load the required packages


## Running applications
#### With Docker
* The design of this setup is to allow for CLI by using the `script/run-digits-rbm` file as the entry point into the application. Given this setup, to run the models call,
    * `$ script/app-env script/run-digits-rbm` plus any command line arguments.
    * The `script/app-env` part puts you into the Docker container's environment, and the `script/run-digits-rbm` part is the entry point into the application as mentioned above.

* You can also run other commands and modules separately inside the Docker container, e.g.
    * `$ script/app-env python some_other_module.py --infile "myfile.txt"` (to run a different Python module)
    * `$ script/app-env python` (to get into the Python shell within the container)

* There is an issue with passing multiple word command line arguments to Docker with this setup even if they are in quotes, just FYI.


#### Without Docker
The design of this setup is to allow for CLI by using the `script/run-digits-rbm` file as the entry point into the application. Given this setup, to run the models call,
  * `$ script/run-digits-rbm` plus any command line arguments.


#### Command line arguments
Whether the models are run with Docker or without, there are a number of optional command line arguments for running the models. The command line arguments are as follows:

##### RBM model
* **n_epochs:** The number of epochs to run (number of times through the data). The default is 1.
* **n_hidden:** The number of hidden nodes to use for the RBM model. The default is 400.
* **batch_size:** The number of samples to use for each mini-batch. The default is 100.
* **digit:** A number in [0, 9]. Samples with target values matching this number are used to train the RBM. The default is 0.
* **visualize:** This boolean argument controls whether the visualizations for the training errors, weights, and free energies are plotted and saved.
* **save_weights:** This boolean argument controls whether the trained weights are saved.

For example (shown with defaults):
```bash
$ script/app-env script/run-digits-rbm  --n_epochs 1
                                        --n_hidden 400
                                        --batch_size 100
                                        --digit 0
                                        --visualize=True
                                        --save_weights=False
```
