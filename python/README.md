# Probabilistic Tensor Decomposition of Neural Population Spiking Activity

[license-img]: https://img.shields.io/badge/license-MIT-green
[license-url]: https://github.com/hugosou/vbgcp/blob/main/LICENSE.md

[python-img]:https://img.shields.io/badge/python-v3.8-blue
[python-url]: https://github.com/hugosou/vbgcp/tree/main/python

[matlab-img]:https://img.shields.io/badge/matlab-R2020-orange
[matlab-url]: https://github.com/hugosou/vbgcp/tree/main/python

[![][license-img]][license-url] [![][python-img]][python-url] 

Python implementations (in development, see [../matlab/](https://github.com/hugosou/vbgcp/tree/main/matlab/)) of [Soulat et al. (2021)](https://arxiv.org/abs/2030.12345)

![alt text](../model_summary.png "Model")

The model (A) decomposes an observed count tensor (eg. binned spikes) using a Negative Binomial distribution that depends on a shape parameter, a constrained offset (B) and low rank tensor (C). 
Variational inference is implemented using a Pólya-Gamma augmentation scheme. 

![alt text](../model_graphical.png "Model")

## Demo


To train the model(s) on the toydataset described in the paper open:
```
python/examples/demo_tensor_variational_inference.ipynb
``` 


