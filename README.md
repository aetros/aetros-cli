# AETROS Python SDK / CLI

<p align="center">
<img src="https://avatars2.githubusercontent.com/u/17340113?v=3&s=200" />
</p>

[![Build Status](https://travis-ci.org/aetros/aetros-cli.svg?branch=master)](https://travis-ci.org/aetros/aetros-cli)
[![PyPI version](https://badge.fury.io/py/aetros.svg)](https://badge.fury.io/py/aetros)

This package is a python application you need to use when you want to train [simple models](http://aetros.com/docu/trainer/models/simple-model)
or if you want to integrate AETROS in your [python model](http://aetros.com/docu/trainer/models/custom-python) using the [AETROS Python SDK](http://aetros.com/docu/python-sdk/getting-started).

### Simple models

It basically retrieves all model information from AETROS, compiles and starts the training, attached with a special logger
callback that sends all information to AETROS Trainer so you can monitor the whole training.

It also contains dataset provider (`aetros.auto_dataset`, with downloader, generator, in-memory iterator and augmentor) for image datasets
which is used if you have a image dataset configured in AETROS Trainer.

### Python models

Please see our documentation [Python SDK: Getting started](http://aetros.com/docu/python-sdk/getting-started).

## Installation

```bash
$ sudo pip install aetros

# update
$ sudo pip install aetros --upgrade
```

### Requirement

For simple models (where we generate the Keras code for you), you need to install Keras 2, Tensorflow and Python 2.7/3.

For custom models (where you start any command and might integrate our Python SDK), you only need Python 2/3.


## Installation development version

If you want to install current master (which is recommended during the closed-beta) you need to execute:

```bash
$ git clone https://github.com/aetros/aetros-cli.git
$ cd aetros-cli
$ make dev-install
$ aetros --help
$ # maybe you have to execute aetros-cli commands using python directly
$ python -m aetros --help
```

To debug issues, you can try to enable debug mode using `DEBUG=1` environment variable in front of the command, example:

```bash
$ DEBUG=1 python -m aetros start owner/model-name/cd877e3f91e137394d644f4b61d97e6ab47fdfde
2017-09-04 17:18:52 osx.fritz.box aetros-job[11153] DEBUG Home config loaded from /Users/marc/.aetros.yml
...
```

You can alternatively to `git clone` download the zip at https://github.com/aetros/aetros-cli/archive/master.zip.
