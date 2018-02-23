## Overview
This repository contains basic RBM and autoencoder example models based on Tensorflow.

## Setup
To run this project in Docker, you will need Docker installed. To install Docker, follow the install directions [here](https://docs.docker.com/install/). It is also possible to run these models without Docker if you have the packages from the [requirements.txt](./requirements.txt) installed along with Python 3.6.

1. Clone repository
2. To create the Docker container, run `$ script/bootstrap`
3. Alternately, if you are not using Docker, make sure you have Python 3 and install the packages from [requirements.txt](./requirements.txt). I would recommend using a [virtual environment](https://virtualenv.pypa.io/en/stable/) if you are not using the Docker container so as not to interfer with your system setup.

## Running applications
### With Docker
* The design of this setup is to allow for CLI by using the `script/run` file as the entry point into the application. Given this setup, to run the models call,
    * `script/app-env script/run` plus any command line arguments.
    * The `script/app-env` part puts you into the Docker container's environment, and the `script/run` part is the entry point into the application as mentioned above.

* You can also run other commands and modules separately inside the Docker container, e.g.
    * `$ script/app-env python some_other_module.py --infile "myfile.txt"` (to run a different Python module)
    * `$ script/app-env python` (to get into the Python shell within the container)

* There is an issue with passing multiple word command line arguments to Docker with this setup even if they are in quotes, just FYI.

### Without Docker
The design of this setup is to allow for CLI by using the `script/run` file as the entry point into the application. Given this setup, to run the models call,
  * `script/run` plus any command line arugments.
