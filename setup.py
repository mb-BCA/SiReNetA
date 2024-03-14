'''setup.py'''

from setuptools import setup, find_packages


with open("requirements.txt") as reqs_file:
    REQS = [line.rstrip() for line in reqs_file.readlines() if line[0] not in ['\n', '-', '#']]

setup(
    name =  'sireneta',
    description = 'A library to study complex networks in the light of canonical propagation models.',
    url =  'https://github.com/mb-BCA/SiReNetA',
    version =  '0.1.dev2',
    license =  'Apache License 2.0',

    author =  'Gorka Zamora-Lopez, Matthieu Gilson',
    author_email =  'gorka@zamora-lopez.xyz',

    install_requires =  REQS,
    packages =  find_packages(exclude=['doc', '*tests*']),
    scripts =  [],
    include_package_data =  True,

    keywords =  'graph theory, complex networks, network analysis, weighted networks',
    classifiers =  [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
