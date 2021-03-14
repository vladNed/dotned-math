from setuptools import setup
from setuptools import find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Deep Learning Math Module'
LONG_DESCRIPTION = 'Math module with formulas and methods to aid in deep learning projects'
SOURCE = 'src'

setup(
   name='dotned',
   author='Vlad Nedelcu',
   author_email='nedelcuvd@gmail.com',
   version=VERSION,
   description=DESCRIPTION,
   long_description=LONG_DESCRIPTION,
   package_dir={"": SOURCE},
   packages=find_packages(where=SOURCE),
   install_requires=['numpy'],
)
