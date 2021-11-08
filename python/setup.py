from setuptools import setup, find_packages

NAME = 'vbgcp'
DESCRIPTION = 'Probabilistic Tensor Decomposition of Neural Population Spiking Activity'
AUTHOR = 'Hugo Soulat'
EMAIL = 'hugo.soulat@gmail.com'
VERSION = "0.1"
URL = 'https://github.com/hugosou/vbgcp'
LICENSE = 'MIT'

install_requires = [
    'numpy',
    'scipy',
    'munkres',
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    install_requires=install_requires,
    python_requires='>=3.7',
)