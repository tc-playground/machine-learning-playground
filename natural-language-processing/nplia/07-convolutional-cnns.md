# Getting Words in Order with Convolutional Neural Networks (CNNs)

---

* Using Neural Networks for NLP
* Finding meaning in word patterns
* Building a Convolutional Neural Network (CNN)
* Vectorizing natural language text in a way that suits neural networks 
* Training a CNN
* Classifying the sentiment of novel text

---

* Language’s true power is not necessarily in the words themselves, but between the words, in the order and combination of words. And sometimes meaning is hidden beneath the words, in the intent and emotion that formed that particular combination of words.

# word order
 * The dog chased the cat.
 * The cat chased the dog.

# Word Proximity
 * " The ship's hull, despite years at sea, millions of tons of cargo, and two mid-sea collisions, shone like new."
 * the word "shone" refers directly to the word "hull". But they are on far ends of the sentence from each other. 
 * how to capture the relationship?
 * spatial patterns
 * temporal patterns

# ANN inputs
 * word vectors
 * one-hot-word vectors.
 * Changing word order on basic ff networks would change the reults.
 * How to deal with sentences of different lengths, or, word of different lengths?

# Convolutional Neural Nets (image example)
 * Convolutional Neural Nets get their name from the concept of sliding (or convolving) a small window over the data sample.
 * mathematics that convolutions appear and they are usually related to time series data.
 * Instead of assigning a weight to each of the elements (say each pixel of an image) of each sample, as in a traditional Feed-Forward net. Instead it defines a set of filters (also known as kernels) that move across the image. A convolution.
 * One key point that was breezed over is the ability of the network to process channels of information.
   * monochrome - one channel per pixel image, 1D tensor.
   * rgb - three channels per per pixel image, 3D tensor.

# Filter sizes (image example)
 * Each filter we make is going to convolve or slide across the input sample (in this case, our pixel values).
 * step-size: the distance traveled during the sliding phase is a parameter.
 * step-size: it is almost never as large as the filter itself.
 * step-size: Each snapshot usually has an overlap with its neighboring snapshot.
 * stide: he distance each convolution "travels" is known as the stride and is typically set to 1.

# Filter compostion (image example)
 * They are composed of two parts:
   * A set of weights (exactly like the weights in the Feed-Forward Neurons from chapter 5)
   * An activation function
 * typically 3x3 (but often other size and shapes)
 * As each filter slides over the image, one stride at a time, it will pause and take "snapshot" of the pixels it is currently covering.
 * The values of those pixels are then multiplied by the weight associated with that position in the filter.
 * The products of pixel and weight (at that position) are then summed up and passed into the activation function (most often this is ReLU).
 * ReLU (recti-linear) activation function.
 * So a single value is produced, that is like a summary of the pixel values under the filter.
 * The output of that activation function is recorded as a positional value in an output "image".
 * After this process. We would have (n) new, 'filtered' images for each filter we defined.

# Dealing with edges (image example)
 * just ignore the fact that the output is slightly smaller.
  * The downfall of this strategy is the data in the edge of the original input is under-sampled as the interior data points are passed into each filter multiple times.
 * padding consists of adding enough data to the outer edges of the input so that the first real data point is treated just as the innermost data points are.
  * The downfall of this strategy is we are adding unrelated data to the input, which in itself can skew the outcome

# Convolutional Pipeline (image example)
 * So we have (n) filters and (n) new images now.
 * The simplest next step is to take each of those filtered images and string them out as input to a feed-forward layer and train with back prop.
 * OR pass these filtered images into a second Convolutional layer with its own set of filters.
  * It turns out the multiple layers of convolutions leads to a path to learning layers of abstractions: first edges, then shapes/colors, and eventually concepts!

 # Learning / Training (image example)
 * NB: Activation function must be differentiable (partial differentiation) in-order to back propgate.
 * A shorthand way of thinking about it is for a given weight in a given filter, the gradient is the sum of the normal gradients that were created for each individual position in the convolution during the forward pass.

# Convolutional Neural Networks for Natural Language Processing by using Word Vectors
 * Use word vector instead of images.
 * focus on the relationships of tokens in one spatial dimension. 
 * Instead of 2 dimensional filter that we would convolve over a 2 dimensional input (a picture) we will convolve 1 dimensional filters over a 1 dimensional input, such as a sentence.
 * our filter shape will be 1 dimensional instead of 2. eg. 1 * 3 word window.
 * Notice each word token (or later character token) is a "pixel" in our sentence "image".
 * The dimension we TIP are referring to when we say 1-dimensional convolution, is the "width" of the phrase
 * The order the "snapshots" are calculated in isn’t important as long as the output is reconstructed in the same way the windows onto the input were oriented.
 * we can take a given filter and take all of its "snapshots" in parallel and compose the output "image" all at once.


# CNN Training Process - Data Perperation.
 * Tokenisation
 * word vectors
 * The maxlen variable holds the maximum length of review we will consider.
  * Pad remaining to 0 or a special pad value.
 * It is helpful to think of the filter sizes, in the FIRST LAYER ONLY, as looking at n-grams of the text.

# Convolutional Neural Network Architecture
 * base Neural Network model class Sequential
 * add is a Convolutional Layer
 * shift (stride) in the convolution will be 1 token. 
 * The kernel (window width) we already set to 3 tokens above.
 * using the 'relu' activation function
 * take our current vector and pass it into a standard feed- forward network, in Keras that is a Dense layer.

# Pooling
 * Pooling is the Convolutional Neural Network’s path to dimensionality reduction.
 * speeding up the process by allowing for parallelization of the computation
 * we make a new "version" of the data sample, a filtered one, for each filter we define
 * The key idea is we are going to evenly divide the output of the each filter into a subsections. 
 * Then for each of those subsections, we will select or compute a representative value. 
 * And then we set the original output of the aside and use the collections of representative values as the input to the next layers.
   * Isn’t throwing away data terrible. Usually, this would not be the best course of action. 
   * But it turns out, this is a path toward learning higher order representations of the source data?
   * In image processing, the first layers will tend to learn to be edge detectors. 
   * While later layers learn concepts like shape and texture, layers after that may learn "content" or "meaning". 
   * Similar processes will happen with text.
 * max-pooling - There are two choices for pooling Average and Max.
   * Average - retains infromation.
   * max - focuses on the most prominent feature.
 * location invariance - dimensionality reduction and the computational savings that come with it, we gain something else special, location invariance.
   * If an element of the the original input is jostled slightly in position in a similar but distinct input sample, the Max Pooling layer will still output something similar.
 * Global Max Pooling - GlobalMaxPooling1D layer
   * instead, of taking the max of a small subsection of each filter’s output, we are taking the max of the entire 21 output for that filter.
 * designed to handle slight variations in word order. 
 
 # Process Summary
  * For each input example we applied a filter (weights and activation function)
  * Convolved across the length of the input, which would output a 1d vector slightly smaller than the original input (1x398 which is input with the filter starting left-aligned and finishing right- aligned) for each filter
  * For each filter output (there are 250 of them, remember) we took the single maximum value from that 1d vector
  * At this point we have a single vector (per input example) that is (1x250 ← the number of filters)

* Now for each input sample we have a 1d vector that the network thinks is a good representation of that input sample. This is a semantic representation of the input.
* once the network is trained, this semantic representation (we like to think of it as a "thought vector") can really be useful.

# Dropout
 * Dropout (represented as layer by Keras, as below) is a special technique developed to prevent overfitting in neural networks. 
 * It is not specific to Natural Language Processing but it does work just as well here.
 * On each training pass, if you "turn off" a certain percentage of the input going to the next layer, randomly chosen on each pass.
 * The model will be less likely to learn the specifics of the training set, "over-fitting", and instead learn more nuanced representations of the patterns.
 * This stops the network over-relying on a particular set of weights (and instead spread the learning over a varyin set of weights).
 * Keras mitigates this in the training phase by proportionally boosting all inputs that are not turned off
   * so the aggregate signal that goes into the next layer is of the same magnitude as it will be during inference stages.

# Output layer Activation Functions
 * 13 available in keras + cusotm functions.
 * binary_crossentropy (single output value)
 * categorical_crossentropy (multiple output values)

# Optimization
 * strategies to optimize the network during training, such as Stochastic Gradient Descent, Adam, RSMProp.
 * alpha - learning rate.

# keras - compile and fit

# keras training
 * compile method
 * fit method
 * callbacks
   * checkpointing - will iteratively save the model only when the accuracy or loss has improved
   * EarlyStopping - will stop the training phase early if the model is no longer improving based on a metric you provide
   * Tensorboard - analysis tool.
* A Keras model can continue the training from this point if it is still in memory, or has been reloaded from a save file. 

# "percentage correct guesses". 
 * This metric is fun to watch but certainly can be misleading, especially if you have a lopsided dataset. 
 * Imagine you have 100 examples, 99 of them are positive examples and only one of them should be predicted as negative. If you predict all 100 as positive without even looking at the data you will still be 99% accurate.

# keras classification
  * predict method

# word channels
 * As our input to the network was a series of words represented as vectors lined up next to each other, 400 (maxlen) words wide x 300 elements long and we used word2vec embeddings for the word vectors. 
 * But there are multiple ways to generate word embeddings as we have seen in earlier chapters. 
 * If we pick several and restrict them to the an identical number of elements we can stack them as we would would picture channel













---

Notes

Where is our chatbot?

? how to deal with multiple sentence lengths over a corpus? When we are looking at word in-order? Dont have to?
  -> Dimensionality reduction to fixed size?
  -> The maxlen variable holds the maximum length of review we will consider.

- wierd flow in intro; moving from tyoes on ANN to ANN inputs.
- The maxlen variable holds the maximum length of review we will consider.