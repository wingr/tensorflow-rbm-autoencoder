#!/usr/bin/env python
import os
import sys
import argparse
import platform

import pytest

from example_model import Example 

def test_setup():
    outfile = './tests/docker_example.txt'
    ex_model = Example(outfile)
    ex_model.run()
    print('Inside Docker using Python version {}'.format(platform.python_version()))
    print('Docker setup correctly, see "Hello World" in {}'.format(outfile))

