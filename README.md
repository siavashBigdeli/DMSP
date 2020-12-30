Caffe and MatCovNet implementations (see [DMSP-tensorflow](https://github.com/siavashBigdeli/DMSP-tensorflow) for TensorFlow V1 and [DMSP-TF2 page](https://github.com/siavashBigdeli/DMSP-TF2) for Tensorflow V2 implementation with super-resolution support)
## Deep Mean-Shift Priors for Image Restoration ([project page](https://www.cs.umd.edu/~zwicker/projectpages/DeepMeanShiftPriors-NIPS17.html))

Siavash Arjomand Bigdeli, Meiguang Jin, Paolo Favaro, Matthias Zwicker

Advances in Neural Information Processing Systems (NIPS), 2017

### Abstract:
In this paper we introduce a natural image prior that directly represents a Gaussian-smoothed version of the natural image distribution. We include our prior in a formulation of image restoration as a Bayes estimator that also allows us to solve noise-blind image restoration problems. We show that the gradient of our prior corresponds to the mean-shift vector on the natural image distribution. In addition, we learn the mean-shift vector field using denoising autoencoders, and use it in a gradient descent approach to perform Bayes risk minimization. We demonstrate competitive results for noise-blind deblurring, super-resolution, and demosaicing.


<img src="https://www.cs.umd.edu/~zwicker/projectpages/DeepMeanShiftPriors-NIPS17-teaser.jpg" alt="Drawing" style="height: 500px;" align="center"/>

See [manuscript](https://arxiv.org/pdf/1709.03749) for details of the method.


This code runs in Matlab and you need to install either
[MatCaffe](http://caffe.berkeleyvision.org) or
[MatConvNet](http://www.vlfeat.org/matconvnet/).
### Contents:

[demo.m](https://github.com/siavashBigdeli/DMSP/blob/master/demo.m): Includes an example for non-blind and noise-blind image deblurring.

[DMSPDeblur.m](https://github.com/siavashBigdeli/DMSP/blob/master/DMSPDeblur.m): Implements MAP function for non-blind image deblurring. Use Matlab's help function to learn about the input and output arguments.

[DAEs](https://github.com/siavashBigdeli/DMSP/tree/master/DAEs): Includes DAE models and function handles (in Caffe and matconvnet).

[data](https://github.com/siavashBigdeli/DMSP/tree/master/data): Includes sample image and blur kernels.

