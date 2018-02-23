"""
Implements a basic example class.
"""

class Example:
    def __init__(self, outfile):
        self.outfile = outfile

    def run(self):
        with open(self.outfile, 'wt') as f:
            f.write('Hello World\n')
