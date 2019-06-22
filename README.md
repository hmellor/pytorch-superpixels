# pytorch-superpixels
Dimensionality reduction allows for the use of simpler networks or more complex objectives. A common way of doing this is to simply downsample the images so that there are fewer pixels to contend with. However, this is a lossy operation so detail (and therefore the upper bound on experimental results) is reduced.

Superpixels slightly alleviate this problem because they are able to encode information about edges within themselves. Generating superpixels is an unsupervised clustering operation. Whilst there are already clustering packages written for Python (some of which this project depends on), they all operate with NumPy arrays. This means that they cannot take advantage of GPU acceleration in the way that PyTorch tensors can.

The aim of this project is to bridge the gap between these existing packages and PyTorch so that superpixels can be readily used as an alternative to pixels in various machine learning experiments.
_______________________________________

This project stems from a module I created for use in my master's thesis.

With some work I think it could be useful for others that wish to utilise superpixels in their Pytorch Machine learning projects.

Ideally I would like to add support for many of the more popular datasets, publish a pypi package and maybe even port this to TensorFlow.

Currently the code is fragmented and unusable, with community support I hope to change that soon.
