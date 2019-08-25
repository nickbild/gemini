# Project Gemini

Project Gemini demonstrates a method to recreate a functionally identical copy of software using deep learning.

The copy operates exactly like the original software, but uses none of the original source code or any proprietary algorithms.  It is de novo self-programmed software.

This method can be used even if the original software has been lost.

## Overview

I have constructed a deep feed-forward neural network that can be trained on an input/output data set from an existing application.  The training process generates a model that recreates the function of, but is implemented in a completely different way than, the original software.

The current implementation will work to recreate deterministic applications (i.e. some set of inputs generates a set of outputs).  The current method would not work for applications with stochastic elements, or if the outputs are dependent on additional sources of state (e.g. saved data on a disk, a hardware timer, etc.).  I think some of these limitations could be overcome with additional research and data sources (e.g. reading disk and memory data into the model), but was not the initial goal of this project, and may be computationally infeasible for a few more years.

## Training

I trained an eight layer feed-forward neural network.  Understanding that a user of the software may not have a vast amount of I/O data from the original software available, I trained on a very modest data set of only 500 inputs.

### AWS

I trained on `g3s.xlarge` instances at Amazon AWS, with Nvidia Tesla M60 GPUs.  The `Deep Learning AMI (Ubuntu) Version 23.1` image has lots of machine learning packages preinstalled, so it's super easy to use.  Just launch an EC2 instance from the web dashboard, then clone my github repo:

```
git clone https://github.com/nickbild/gemini.git
```

Then start up a Python3 environment with PyTorch and dependencies and switch to my codebase:

```
source activate pytorch_p36
cd gemini
```

Now, run my training script:

```
python3 train.py
```

## Media

See it in action:
[YouTube](https://www.youtube.com/watch?v=kNbbeXuxwkA)

## About the Author

[Nick A. Bild, MS](https://nickbild79.firebaseapp.com/#!/)
