from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='survey_analysis',
    version='0.1',
    packages=find_packages(where='src', include=['survey_analysis']),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
)