"""
Unit and regression test for the vccfcoords package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import vccfcoords
import numpy
import pandas
import tqdm
import requests
import matplotlib
import seaborn
import scipy

def test_package_imported():
    """Test, will always pass so long as import statement worked."""
    assert "autopartonomy" in sys.modules
    
def test_package_libaries_imported():
    """Test, if the import statements of package libraries worked."""
    assert "numpy" in sys.modules
    assert "pandas" in sys.modules
    assert "tqdm" in sys.modules
    assert "requests" in sys.modules
    assert "matplotlib" in sys.modules
    assert "seaborn" in sys.modules
    assert "scipy" in sys.modules

    
    
