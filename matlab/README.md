# Probabilistic Tensor Decomposition of Neural Population Spiking Activity
## Matlab implementation

[license-img]: https://img.shields.io/badge/license-MIT-green
[license-url]: https://github.com/hugosou/vbgcp/blob/main/LICENSE.m


[python-img]:https://img.shields.io/badge/python-v3.8-blue
[python-url]: https://github.com/hugosou/vbgcp/tree/main/python

[matlab-img]:https://img.shields.io/badge/matlab-R2020-orange
[matlab-url]: https://github.com/hugosou/vbgcp/tree/main/python

[![][license-img]][license-url] [![][matlab-img]][matlab-url] 


## Requirements
- MATLAB R2020a

- Laurent S (2021). Khatri-Rao product (https://www.mathworks.com/matlabcentral/fileexchange/28872-khatri-rao-product), MATLAB Central File Exchange. Retrieved May 28, 2021. 

- MunkRes:  Yi Cao (2021). Munkres Assignment Algorithm (https://www.mathworks.com/matlabcentral/fileexchange/20328-munkres-assignment-algorithm), MATLAB Central File Exchange. Retrieved May 28, 2021. 

- (G)CP baselines
Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox Version 2.6, Available online, February 2015. URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.  GCP uses default optimization method is L-BFGS-B: https://github.com/stephenbeckr/L-BFGS-B-C

## Demo

To train the model(s) on the toydataset described in the paper, run:

```
demo_tensor_variational_inference.m
```
PG approximation Figures can be generated with: 

```
study_polyagamma.m
```
