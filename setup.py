from setuptools import setup
from setuptools import find_packages

setup(name='aetros',
      version='0.1.0',
      description='Deep Learning end-to-end application: Worker',
      author='Marc J. Schmidt',
      author_email='marc@marcjschmidt.de',
      url='https://github.com/aetros/aetros-cli',
      download_url='https://github.com/aetros/worker/tarball/0.1.0',
      license='MIT',
      install_requires=(
            'keras>=1.0.6',
            'requests',
            'numpy',
            'scipy',
            'h5py',
            'py-cpuinfo==0.2.3'
      ),
      packages=find_packages())