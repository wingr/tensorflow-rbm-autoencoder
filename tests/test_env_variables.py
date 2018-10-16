import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.utils.data_utils import DataUtils  # noqa


class TestEnvironmentVariables(object):
    """Class to test contents of .env file within script/ directory."""

    def test_hadoop_host_existence(self): 
        """Fail if HADOOP_HOST not found."""
        host = os.environ['HADOOP_HOST']
        assert len(host) > 1

    def test_hadoop_host_type(self):
        """Fail if HADOOP_HOST not string."""
        host = os.environ['HADOOP_HOST']
        assert isinstance(host, str)

    def test_hadoop_port_existence(self): 
        """Fail if HADOOP_PORT not found."""
        port = int(os.environ['HADOOP_PORT'])
        assert port > 0

    def test_hadoop_port_type(self):
        """Fail if HADOOP_PORT not int after casting."""
        port = int(os.environ['HADOOP_PORT'])
        assert isinstance(port, int)

    def test_hadoop_user_existence(self): 
        """Fail if HADOOP_USER not found."""
        username = os.environ['HADOOP_USER']
        assert len(username) > 0

    def test_hadoop_user_type(self):
        """Fail if HADOOP_USER not string."""
        username = os.environ['HADOOP_USER']
        assert isinstance(username, str)

    def test_hadoop_user_not_deploy(self):
        """Fail if HADOOP_USER is deploy."""
        username = os.environ['HADOOP_USER']
        assert username.lower() != 'deploy'

    def test_presto_host_existence(self): 
        """Fail if PRESTO_HOST not found."""
        host = os.environ['PRESTO_HOST']
        assert len(host) > 1

    def test_presto_host_type(self):
        """Fail if PRESTO_HOST not string."""
        host = os.environ['PRESTO_HOST']
        assert isinstance(host, str)

    def test_presto_port_existence(self): 
        """Fail if PRESTO_PORT not found."""
        port = int(os.environ['PRESTO_PORT'])
        assert port > 0

    def test_presto_port_type(self):
        """Fail if PRESTO_PORT not int after casting."""
        port = int(os.environ['PRESTO_PORT'])
        assert isinstance(port, int)

    def test_presto_driver_existence(self): 
        """Fail if PRESTO_DRIVER not found."""
        driver = os.environ['PRESTO_DRIVER']
        assert len(driver) > 0

    def test_presto_driver_type(self):
        """Fail if PRESTO_DRIVER not string."""
        driver = os.environ['PRESTO_DRIVER']
        assert isinstance(driver, str)
