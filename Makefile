#!/usr/bin/env make

default:
	@echo "There is no default action."

install:
	python3 -m pip install --user -r requirements.txt
	python3 -m pip install --user .

uninstall:
	python3 -m pip uninstall vivisection


test-deploy:
	@rm -fR build dist
	@python3 setup.py sdist bdist_wheel --universal && twine upload -r pypitest dist/*
	@pip3  install --user TMO4CT --index-url https://test.pypi.org/simple/
	@pip3 uninstall TMO4CT

deploy:
	@rm -fR build dist
	@python3 setup.py sdist bdist_wheel --universal && twine upload -r pypi dist/*

format:
	@find . -name "*py" -exec autopep8 -i {} \;
