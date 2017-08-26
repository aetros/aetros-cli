zip:
	rm -f aetros-cli.zip
	zip aetros-cli.zip aetros/*.py aetros/commands/*.py README.md setup.cfg setup.py

dev-install:
	python setup.py sdist
	python -m pip install --upgrade dist/aetros-`python -m aetros --version`.tar.gz