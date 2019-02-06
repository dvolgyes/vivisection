#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from vivisection import (__description__, __version__, __author__,
                         __email__, __summary__, __license__)


requirements = ['torch', 'numpy', 'termcolor', 'loguru', 'pygments']

setup_requirements = requirements

test_requirements = requirements

setup(
    author=__author__,
    author_email=__email__,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description=__summary__,
    install_requires=requirements,
    license=__license__,
    long_description=__description__,
    include_package_data=False,
    keywords=['debug', 'pytorch'],
    name='vivisection',
    packages=find_packages(where='.'),
    scripts=[],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.uio.no/davidvo/vivisection',
    version=__version__,
    zip_safe=False,
    python_requires='>=3.6',
)
