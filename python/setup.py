from setuptools import setup

install_requires = [
    'numpy',
    'scipy',
    'munkres'
]

setup(name="vbgcp",
      version="0.0.1",
      description="Probabilistic Tensor Decomposition of Neural Population Spiking Activity",
      author="Hugo Soulat",
      url="https://github.com/hugosou/vbgcp/tree/main/python",
      packages=["vbgcp", "examples"],
      install_requires=install_requires,
      license="MIT")

