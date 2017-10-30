from setuptools import setup
from setuptools import find_packages
import aetros.const

setup(name='aetros',
      version=aetros.const.__version__,
      description='Python SDK for aetros.com',
      author='Marc J. Schmidt',
      author_email='marc@marcjschmidt.de',
      url='https://github.com/aetros/aetros-cli',
      download_url='https://github.com/aetros/aetros-cli/tarball/' + aetros.const.__version__,
      license='MIT',
      entry_points={
          'console_scripts': ['aetros=aetros:main'],
      },
      install_requires=('requests',
                        'numpy',
                        'h5py',
                        'coloredlogs',
                        'psutil>=5.3.0',
                        'ruamel.yaml>=0.15.0',
                        'cherrypy>=7.1.0',
                        'Pillow>=4.0.0',
                        'paramiko>=2.3.1',
                        'py-cpuinfo>=3.3.0',
                        'msgpack-python>=0.4.8'),
      packages=find_packages())
