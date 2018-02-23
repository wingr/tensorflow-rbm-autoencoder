## Overview
This repository contains basic RBM and autoencoder example models based on Tensorflow.

## Setup
1. Clone repository
2. To create the Docker container, run `$ script/bootstrap`

## Running applications
* The design of this setup is to allow for CLI by using the `script/run` file as the entry point into the application. Given this setup, to run the models call,
    * `script/app-env script/run` plus any command line arguments.
    * The `script/app-env` part puts you into the Docker container's environment, and the `script/run` part is the entry point into the application as mentioned above.

* You can also run other commands and modules separately inside the Docker container, e.g.
    * `$ script/app-env python some_other_module.py --infile "myfile.txt"` (to run a different Python module)
    * `$ script/app-env python` (to get into the Python shell within the container)

* There is an issue with passing multiple word command line arguments to Docker with this setup even if they are in quotes, just FYI.
