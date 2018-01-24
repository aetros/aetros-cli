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

      packages=find_packages(),
      install_requires=['requests',
                        'numpy',
                        'coloredlogs',
                        'psutil>=5.3.0',
                        'ruamel.yaml>=0.15.0',
                        'cherrypy>=7.1.0',
                        'six>=1.11.0',
                        'Pillow>=4.0.0',
                        'paramiko>=2.3.1',
                        'terminaltables>=3.1.0',
                        'colorclass>=2.2.0',
                        'docker>=2.7.0',
                        'simplejson>=3.13.2',
                        'py-cpuinfo>=3.3.0',
                        'msgpack-python>=0.4.8']
)
