from setuptools import setup, find_packages

with open('recommendations.txt') as f:
    recommendations = f.read().splitlines()

setup(
    name='sklearn-mrmr',
    version='0.1',
    packages=find_packages(),
    description='sklearn-compatible implementation of mRMR feature selection',
    author='Benjamin Horvath',
    author_email='benhorvath@gmail.com',
    url='https://github.com/benhorvath/sklearn-mrmr/',
    install_requires=recommendations,
)