# Improving Retention with Long Short- Term Memory Networks (LSTMs)

----

* Adding deeper memory to Recurrent Neural Nets • Gating information inside Neural Nets
* Classification vs. Generation
* Modeling Language Patterns

----

# RNN short-comings
 * For all the benefits Recurrent Neural Nets provide for modeling relationships in time series data, and therefore possibly causal relationships, they suffer from one main deficiency: the effect of a token is almost completely lost by the time two time steps have passed.
 * Any effect the first node will have on the third node (the 2nd time step thereafter), will be thoroughly stepped on by new data introduced in the intervening time step.
 * This is, of course, important to the basic structure of the net, but it precludes the common case in human language that the tokens may be deeply interrelated but be separated greatly in the sentence.
 * For example:
  * The young woman went to the movies with her friends.
  * The young woman, having found a free ticket on the ground, went to the movies. (hard for RNNs)

# What we need is a way to remember the past across the entire input. The advent of Long Short-Term Memory (and the closely related Gated Recurrent Unit) is the tool developed for just this sort of case.

---

# Long Short-Term Memory network (LSTM)
 * introduce the concept of a state for each layer in the Recurrent Network.
 * The state acts as its memory.
 * You can think of it as adding attributes to a class in Object-Oriented Programming. 
 * The attributes of the memory state are updated with each training example.
 * The magic of LSTMs lies in the fact that the rules that govern what information is stored in the state are trained neural nets themselves. 
 * They learn along with the rest of the Recurrent Net!
 * With the introduction of a short term memory we can begin to learn dependencies that stretch not just one or two tokens away, but across the entirety of each data sample.
 * we can actually start to generate novel text around those pattern.

# LSTM Memory Unit - LSTM cell
 * That state is affected by the input and affects the output of the layer just like happens in a normal recurrent net.
 * but that state persists across all time steps of the time series.
 * Standard RNN
   * add a memory state that passes through time steps of the network, so that each time step iteration has access to it.


# LSTM cell
 * A cell can be composed of one or more output neurons, so the corollary to an LSTM cell is definitely closer to a Neural Network layer, rather than the neurons themselves.
 * Each cell instead of being a series of weights on the input and an activation function on those weights, is now somewhat more complicated.
 * the input to the layer (or cell) is a combination of the input sample and output from the previous time step.
 * As information flows into the cell instead of a vector of weights, it is now is greeted by three gates:
   * Forget gate
   * Input/Candidate gate 
   * Ouput gate
 * Each of these gates is a feed-forward network layer composed of a series of weights that the network will learn and an activation function.
   * Technically one of the gates is composed of two feed- forward networks, so there will actually be four sets of weights to learn in this layer. 
 * The weights and activations will aim to allow information to flow through the cell in different amounts all in relation to the state (or memory) of the cell.

 # What is the Memory?
  * The memory is going to be represented by a vector that is the same number of elements as neurons in the cell. 
  * Our example has a relatively simple 50 neurons, so the memory unit will be a vector of floats that is 50 elements long.

# Now what are these gates? 
 * Our "journey" through the cell is not a single road, there are branches and we will follow each for a while then back up, progress, branch, and finally come back together for the grand finale of the cell’s output.
 * We take the first token from the first sample and pass its 300 element vector representation into the first LSTM cell. 
 * On the way in, it is concatenated with the vector output from the previous time step, which is a 0 vector in the first time step. 
   * In this example we will have a vector that is 300 + 50 elements long. 
 * Sometimes you will see a 1 appended to the vector, this would correspond to the bias term. 
 * # Forget Gate - At the first fork in the road, we hand off a copy of the combined input vector to the ominous sounding forget gate.
   * The goal of the forget gate is to learn, based on a given input, how much of the cell’s memory do we want erase
     * In some cases we want to remember e.g. gender based noun; and sometimes forget, e.g. noun switches.
     * we are trying to model not just long-term dependencies within a sequence but crucially forget long-term dependencies as new ones arise
     * hence the importance of the forget gate.
  * the network isn’t working in these kind of explicit representations, 
  * it is trying learn a set of weights that when applied to all of the input from a certain sequence it updates the memory and hence the output appropriately.
  * The activation function for a forget gate is the sigmoid function, as we want the output for each neuron in the gate to be between 0 and 1.
  * The output vector of the forget gate is then a "mask" of sorts, albeit a porous one that "erases" elements of the memory vector.
  * As the forget gate outputs values closer to 1, more of the memory’s knowledge in the associated element is retained for that time step. 
  * Closer to 0 and more of the value of that element is erased.
* # Candidate Gate - Remember things
  * The candidate gate is the gate mentioned above with 2 separate networks inside. 
    * The first is a net with a sigmoid activation function whose goal is to learn which values of the memory register to update.
    * This very closely resembles the mask generated in the forget gate.
    *  The second gate determines what values we are going to update the memory with. 
    * This second part has a tanh activation function so each output value will be between -1 and 1. 
    * The output of these two are multiplied together element-wise. 
    * The resulting vector from this multiplication is then added, again element-wise, to the memory register, thus remembering new details.
    * This gate is learning simultaneously which values to extract and the magnitude of those particular values
    * The mask and magnitude become what is added to the memory state. 
    * As in the forget gate, the candidate gate is learning to mask off the "inappropriate" information before adding into the cell’s memory.
 * # Output Gate - as old, seemingly irrelevant, things are forgotten and new things are remembered, we arrive at the the last gate of the cell; the output gate.
   * Up until this point in the journey through the cell, we have only written to the cell’s memory. 
   * Now it is finally time to get some use out of this structure. 
   * The output gate takes the input (remember this is still the concatenated input a time step t and the output of the cell from time step t-1) and passes it into the output gate.
   * The concatenated input is passed into the weights of the n neurons then we apply a sigmoid activation function outputting an n-dimensional vector of floats, just like the output of a simpleRNN. 
   * But instead of handing that information out through the cell wall, we pause.
   * The memory structure we have built up is now primed and it gets to weigh in on what we should output. 
   * This judgment is achieved by using the memory to create one last mask.
   * The mask created from the memory is just the memory state with a tanh function applied element- wise. 
   * This gives an n-dimensional vector of floats between -1 and 1. 
   * That mask vector is then multiplied element-wise against the raw vector computed in the output gate’s first step. 
   * The resultant n-dimensional vector is finally passed out of the cell as the cell’s official output at time step t.
   * Remember the output from an LSTM cell is just like the output from a Simple Recurrent Neural Network layer.
    * It is passed out of the cell as the layer output (at timestep t) and to itself as part of the input to time step t+1.

# LSTM - Backpropagation
 * A vanilla RNN is susceptible to a vanishing gradient (small weights) / exploding grdaient (large weights).
   * because the derivative at any given time step is a factor of the weight themselves.
 * An LSTM avoids this problem via the memory state itself.
 * The derivatives and back propgation are local to the LSTM cells.
 * The error of the entire function is this way kept "nearer" to the neurons for each time step. This is known as the error carousel.

 # In practice - fixed length.
  * Keras SimpleRNN Layer for the Keras LSTM .
  * Tokenize the text and embed those using word2vec. (word to vec is a trained ANN to transform tokenised text into BoVs) 
  * Then we will pad/truncate the sequences again to 400 tokens each using the functions we defined in the previous chapters.

* You can start to see how much gain there is to be had in providing the model with a memory when the relationship of tokens is so important.

* How do you model humor, sarcasm, angst? Can they be modeled together?

---

# Dirty Data
 * Fixed size data: feed-forward layer at the end of the chain and those require vectors of fixed size. 
 * Our implementations of Recurrent Neural Nets, both simple and LSTM, are striving toward a thought vector that we can pass into a feed-forward layer for classification. 
 * So that the thought vector is of consistent size we have to unroll the net a consistent number of time steps. 
 * With all the power of neural nets and their ability to learn complex patterns, it is easy to forget a neural net is in most cases just as good at learning to discard noise.
 * Still subject to skew though (which is related to context).
 * If your training set is composed of documents thousands of tokens long, you may not get a very accurate classification of something only 3 tokens long padded out to 1000. 
 * Careful of dropping terms:
   * "I don’t like this movie." vs I like this movie."
* There are two more commonplace approaches that provide decent results without exploding the computational requirements. 
* Both involve replacing the unknown token with a new vector representation. 
  * for every token not modeled by a vector, randomly select a vector from the existing model and use that instead.
  * replace all tokens not in the word vector library with a specific token, usually referenced as "UNK".
    * For padding use "PAD".

---

# Letters and meaning - atomicity.
 * Words have meaning, we can all agree on this. It only seems natural then to model natural language with these basic building blocks.
 * But, of course, words aren’t atomic at all. 
 * They are made up of smaller words, stems, phonemes. 
 * But they are also even more fundamentally, a sequence of characters.
 * To notice a repeated suffix after a certain number of syllables, which would quite probably rhyme may be a pattern that carries meaning, perhaps joviality or derision.
 * And there’s not so many letters to deal with! So we have less variety of input vectors to deal with.
 * The patterns and long term dependencies found at the character level can vary greatly across voices.
 * expensive compared to work level LTSTM and CNNs.
 * It turns out that the character-level model can be extremely good at modeling the language itself given a broad enough example.
 * Or model a specific kind of language given a focused enough training set, say from one author instead of thousands. 
 * Either way this is a first step toward generating novel text with a Neural Net.

 # Generative ANNs
  * generate new text with a certain "style" or "attitude"
  * Much like a Markov Chain which predicts the next word in a sequence based on the probability of any given word appearing after the 1-gram or 2-gram or n-gram that just occurred, our LSTM model can learn the probability of the next word based on what it just saw, but with the added benefit of memory!
  * A Markov chain only has information about the n-gram is using to search and the frequency of words that occur after that n-gram. 
  * The RNN model does something very similar in that it encodes information about the next term based on the few that preceded it. 
  * With the LSTM memory state, the model has a greater context in which to judge the most appropriate next term. 
  * And most excitingly, we can predict the next character based on the characters that came before. 
  * This level of granularity is beyond a basic Markov Chain.

# Training Generatives LTSTMs
 * First we are going to abandon our classification task.
 * The real core of what the LSTM is learned in in the LSTM cell itself.
 * But as earlier we were using the models successes and failures around a specific classification task to learn that is not necessarily modeling a more general representation of language in general.
 * So instead of using the sentiment label of the training set as the target for learning, we use the training samples themselves!
 * For each character in the sample we want to learn to predict the next character.
 * This can work on the word level, but we are going to cut to the chase and go straight down to the character level with the same concept.
 * Instead of a thought vector coming out of the last time step we are going to focus on the output of each time step individually.
 * The error will still backpropagate through time from each time step back to the beginning but the error is determined specifically at the time step level.
 
 * first thing we need to is adjust our training set labels. 
 * The output vector will be measured not against a given classification label but against the one-hot encoding of the next character in the sequence.
 * We can also fall back to a simpler model and instead of trying to predict every next character, just predict the next character for a given sequence.
   * Keras: drop eturn_sequences=True configuration.
 * learn to predict the 41st character given what came before.
   * no dropout. As we are looking to specifically model this dataset, we have no interest in generalizing to other problems so not only is over-fitting okay, it is ideal.
* Since the output vectors are 50 dimensional vectors describing a probability distribution over the 50 possible output characters we can sample from that distribution.
* Given a chracter:
  * By looking at the highest value in the output vector, we can see what the network thinks has the highest probability of being the next character. 
  * In explicit terms, the index of the output vector with the highest value (which will be between 0 and 1) will correlate with the index of the one-hot encoding of the expected token.
* Since the last layer in the network is a softmax, the output vector will be probability distribution over all possible outputs of the network.
*  But here we aren’t looking to exactly recreate what the input text was just what is likely to come next. 
* Just as in a Markov chain, the next token is selected randomly based on the probability of the next token, not the most commonly occurring next token.
* # temperature (or diversity )
  * less than 1 will tend toward a more strict attempt to recreate the original text
  * while a temp greater than 1 will produce a more diverse outcome, 
* We are taking a random chunk of 40 (maxlen) characters from the source and predicting what will come next character by character.
*  We then append that to the input sentence, drop the first character and predict again on that new subset of 40. 

---

# LSTM Varaitaions - Gated Recurrent Unit (GRU)
 * combines the forget gate and the candidate choice branch from the candidate gate into a single update gate.
 * This saves on the number of parameters to learn and has been shown to be comparable to a standard LSTM while being that much less computationally expensive.
 * peephole connections
   * The idea is that each gate in a standard LSTM cell has access to the current memory state directly, taken in as part of its input.
   * the gates contain additional weights of the same dimension as the memory state. 
   * The input to each gate is then a concatenation of the input to the cell at that time step and the output of the cell from the previous time step and the memory state itself

# Stacked LSTM Layers
 * Have LSTM stacked layers of LSTMS,

---

# Summary
 * Remembering information in sequential inputs is possible
 * It’s important to forget information that is no longer relevant
 * Only some the new information needs to be retained for the upcoming input
 * All of the "rules" around remembering and forgetting can be learned
 * If we can predict what comes next, we can generate novel text from probabilities

---

* from nltp corpus import gutenberg
* keras example for LTST Niestche generative test example.




---

Notes

+ Good intutition on character based training.

- No definition on Markov chain generationn.

    