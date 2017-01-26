from setuptools import setup
from setuptools import find_packages
import aetros

setup(name='aetros',
      version=aetros.const.__version__,
      description='Python SDK for aetros.com',
      author='Marc J. Schmidt',
      author_email='marc@marcjschmidt.de',
      url='https://github.com/aetros/aetros-cli',
      download_url='https://github.com/aetros/aetros-cli/tarball/' + aetros.const.__version__,
      license='MIT',
      entry_points={
          'console_scripts': ['aetros = aetros:main'],
      },
      install_requires=('keras>=1.0.6',
                        'requests',
                        'numpy',
                        'scipy',
                        'h5py',
                        'psutil',
                        'image',
                        'cherrypy>=7.1.0',
                        'py-cpuinfo==0.2.3'
                        ),
      packages=find_packages())
