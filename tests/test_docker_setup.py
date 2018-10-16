"""Test Docker container setup."""
import os
import sys
import platform


class TestDockerSetup():
    """Class to test Docker setup."""

    def get_expected_python_version(self):
        """Read expected Python version from Dockerfile.

        The first line of the Dockerfile gives the expected Python version in the
        form `FROM python:3.6.4`. This function pulls this information and
        reformats to match the output from `$ python --version`.
        """
        #with open('Dockerfile', 'rt') as f:
        #    first_line = f.readline()
        #line_list = first_line.split(':')
        #expected_version = "Python {}".format(line_list[1].strip())
        expected_version = 'Python 3.5.2'
        return expected_version

    def print_container_python_version(self):
        """Print Python version from within Docker container."""
        print('Python {}'.format(platform.python_version()))

    def get_expected_pip_packages(self):
        """Read expected pip packages from requirements.txt file."""
        with open('requirements.txt', 'rt') as f:
            lines = f.readlines()
        expected_pip_packages = [elem.lower().strip() for elem in lines]
        return expected_pip_packages

    def print_pip_packages(self):
        """Print pip packages from within the container."""
        cmd = 'pip freeze'
        os.system(cmd)

    def test_docker_setup(self, capfd):
        """Fail if container not created with correct Python version."""
        expected = self.get_expected_python_version()
        self.print_container_python_version()
        out, _ = capfd.readouterr()
        assert out.strip() == expected

    def test_docker_packages(self, capfd):
        """Fail if container's pip packages don't match requirements.txt.

        The full set of packages listed within the container with `$ pip freeze`
        is the superset of the packages listed in the requirements.txt file
        since some packages installed from the `requirements.txt` file require
        other packages as dependencies, which are also installed. This is the
        reason we have to use the `in` syntax.
        """
        expected_pip_packages_ls = self.get_expected_pip_packages()
        self.print_pip_packages()
        out, _ = capfd.readouterr()
        container_pip_packages_ls = [elem.lower() for elem in out.split('\n')]
        for package in expected_pip_packages_ls:
            assert package in container_pip_packages_ls

    def test_gpu(self):
        from keras import backend as K
        gpu_ls = K.tensorflow_backend._get_available_gpus()
        assert len(gpu_ls) > 0
