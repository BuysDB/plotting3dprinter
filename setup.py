#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))
version = {}
with open(os.path.join(_here, 'plotting3dprinter', 'version.py')) as f:
    exec(f.read(), version)

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.md')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(

    name='plotting3dprinter',
    version=version['__version__'],
    description='Generate gcode to plot on a 3d printer with a pen',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Buys de Barbanson',
    author_email='aegit@buysdb.nl',
    url='https://github.com/BuysDB/plotting3dprinter',

    license='MIT',
    packages=[
    'plotting3dprinter',
        ],


    entry_points={'console_scripts': ['svgto3dprintplot=plotting3dprinter.svgto3dprintplot_cmd:main']},

    scripts=[


        ],

  install_requires=[
       'numpy>=1.16.5','lxml'
   ],
    #setup_requires=["pytest-runner"],
    #tests_require=["pytest"],

    include_package_data=True,

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
         'License :: OSI Approved :: MIT License'
        ]
)
