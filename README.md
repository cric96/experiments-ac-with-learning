# Experiments of Aggregate Computing combined with AI 

## Hop-count with variable input

### Description:

In this branch, I try to use AI models to learn at extracting the correct value within a node neighbourhood. Even in this case, I try to learn [Hop Count Function](https://github.com/cric96/experiments-ac-with-learning/tree/hop-count-regression), so I recommend you to read that part first.

In the following, there is a brief description of regression models that can handle variable input size.
#### Background

In general, regression models (and also classification models) presume that the input size is fixed. It is a problem in our domain: we don't have any guarantee in the input size (i.e. the neighbour values that the node gather) since the network is not *fixed* and can change over time.
So, as the first experiments, I try to use two regression models that are independent of the input size: [(Fully) Convolutional Neural Network](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks), and [Recurrent Neural Network](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks). I want to point out that in general is a good practice to pre-process input, so even the previous approaches could be valuable (e.g. extracting only the min or a set of relevant values, in ML is usually to select a group of relevant feature).

#### Convolutional Neural Network (CNN)
CNN are neural network models used to process image data. Indeed they are inspired by the pioneering works in the neural cortex about simple cells and complex cell. The idea is that at an input, we apply a sequence of [Convolutional filter](https://en.wikipedia.org/wiki/Kernel_(image_processing)) invariant for translation and rotation. These operations, applied multiple times, found relevant features. These are then given in input to a standard Multi-Layer Perception (or another classification/regression model) that can learn easier the association input/output. This is an example of Feature Learning, i.e. the neural network learns what features should extract in order to understand the right input and output association. 

A pictorial representation of this network could be:
![image](https://stanford.edu/~shervine/teaching/cs-230/illustrations/architecture-cnn-en.jpeg?3b7fccd728e29dc619e1bd8022bf71cf)

CNN (and also FCNN) are also used with one dimensional signal (i.e. input sequence or audio) as in [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
#### Fully Convolutional Neural Network (FCNN)
FCCN are CNN without the last dense layers. Namely 
>They employ solely locally connected layers, such as convolution, pooling and upsampling. Avoiding the use of dense layers means less parameters (making the networks faster to train). It also means an FCN can work for **variable** image sizes given all connections are local.

FCNN are also used for classification and regression. In these case, a [Global pooling layer](https://paperswithcode.com/method/global-average-pooling#:~:text=Global%20Average%20Pooling%20is%20a,in%20the%20last%20mlpconv%20layer.) is required to flatten the multi-dimensional input shape to a "flat"-dimensional shape.

#### Recurrent Neural Network (RNN)
The aforementioned neural network is part of "feed-forward" models, i.e. the data flow to the input layer into the output layer without having a feedback loop on some layer. RNNs instead, introduce a feedback loop for each layer. In this way, it is possible to add a short of *memory*: the output at the time step *t* depends upon the output at the state *t-1*. Another feature that has these network is that is input invariant size. In fact, they are used for *sequential* and *time-series* data.
An RNN can be summarized as follow:
![image](https://stanford.edu/~shervine/teaching/cs-230/illustrations/description-block-rnn-ltr.png?74e25518f882f8758439bcb3637715e5)

### Validation configuration

### What happens

![Result](assets/plot/model-comparison.png)

### Final remarks
