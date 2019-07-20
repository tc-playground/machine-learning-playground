# Sequence to Sequence Models and Attention Mechanism

---

* Mapping relationships amongst text 
* Using NLP for translation
* Developing language models for chat 
* Paying Attention to what we hear

---

# Sequence to Sequence networks - 0
 * sequence-to-sequence networks have two components:
   * # sequence encoder: is a network which turns the text into a vector representation.
   * # sequence decoder: is a ngerative model that acts on the vector representation.
 * A sequence decoder can turn a vector back into human readable text again. 
 * The resulting text can the answer to a question.
 * The resulting text can be the German translation of the English input text.
 * leveraging the classification aspect of the LSTM we can create a thought vector 
 * and use that as the input to a second different LSTM that only tries to predict token by token (as in the latter example.) 
 * And thus we have a way to map a sequence of input to a distinct sequence of output.


 # LSTM Networks 
  * we introduced a way for the network to use that information as it sees the next piece of input. 
  * The input at each time step affects the memory via the forget and update gates, 
  * but the output of the network that time step is dictated not solely by the input but by a combination of the input and the current state of the memory unit.

# word-vector

# thought-vector

# sequence-to-sequence networks - seq2seq
 * develop a translation model to translate texts from English to German
 * Long-short-term-memory networks:  we can map sequences of characters or embedded words to another.
 * We previously discovered how we can predict an element at time step t based on the previous element at the time step t-1.
 * but, the input and output sequences need to have the same sequence lengths.
 * the likelihood of a sequence length in one language will match the sequence length in another language is fairly low.
 * seq2seq, solve this limitation of language modeling concepts by creating an input representation in the form of a thought vector, and then using that thought (or sometimes called context) vector to generate the output sequence.
 * A thought vector is very similar to a word vector. 
 * The network will find a representation with a fixed length to represent the content of the input. 
 * thought vectors are used to determine a numerical representation of a thought of any input length. (Geofrey Hinton).
 *  sequence-to-sequence network consists out of two recurrent neural networks.
   * Encode -> Decoder
* encoder, is turning the input text (e.g. a user question) into the thought vector.
  * we are ignoring the output of the encoder network and only use the state of the Recurrent Neural Network as the thought vector.
* thought vector will serve as initial state of the decoder network
* The decoder entowrk then uses the initial state and a start token to generate the first sequence element of the decoder (e.g. a character or word).
* This generated sequence element will then become the input to the decoder to generate the next element.
* and so on until the maximum number of sequence elements is reached or an end-of-sequence token is generated.
* That way the decoder will turn the thought vector into a fully decoded response to the initial input sequence (e.g. the user question)
* Splitting the solution into two networks with the thought vector as the binding piece in between allows us to map input sequences to output sequences of different lengths.

# Autoencoder?
 * A seq2seq network takes a sequence, encodes it into a thought vector and then decodes it into an output sequence.
 * While the structure looks very similar, autoencoders are different.
 * Their purpose is to find the most optimal vector representation of input data, so that it can be reconstructed by the networks decoder with the lowest loss.
 * The seq2seq network purpose is to find a dense vector representation of the input data (e.g. an image or text) which allows the decoder to reconstruct it with the smallest error.
 
---

 # teacher-forcing training method for chatbots

---

 # reduce training complexity using bucketing
  * Too much padding can make the computation expensive, especially when the majority of the sequences are short and only a handful of them use close to the maximum length of tokens.
  * Bucketing can reduce the computation in these cases.
  * you can sort the sequences by length and use different sequence lengths during different batch runs.


# attention mechanism
 * As with Latent Semantic Analysis, longer input sequences (documents) tend to produce thought vectors that are less precise respresentations of those documents.
 * A thought vector is limited by the dimensionality of the LSTM layer (the number of neurons).
 * imagine the case when you want to train a seq2seq model to summarize online articles. 
 * In this case, your input sequence can be a lengthy article which should be compressed into a single thought vector to generate e.g. a headline.
 * As you can imagine, it is tricky to train the network to determine the most relevant information in that longer document. 
 * A headline or summary (and the associated thought vector) must be focused on a particular aspect or portion of that document rather than attempt to represent all of the complexity of the meaning of that document.
 * the idea is to tell the decoder what to pay attention to in the input sequence.
 * This "sneak preview" is achieved by allowing the decoder to look at the states of the encoder network in addition to the thought vector. 
 * In other words, the attention mechanism allows a direct connection between the output and the input by selecting relevant input pieces.
 * With the attention mechanism, the decoder receives an additional input with every time representing the one (or many) encoder input time steps to pay "attention" to, at this given decoder time step. 
  * All states from the encoder will be represented as a weighted average for each decoder time step.

---

# seq2seq applications
 * Seqence-to-Sequence networks are well suited for any machine learning application with variable length input sequences or variable length output sequences.
 * aaplications areas:
   * Chat bot conversations 
   * Question-answering
   * Machine translation
   * Image captioning
   * Visual question answering 
   * Document summarization
 * Deep Mind's Q and A datasets
 * When you need your dialog system to respond reliably in a specific domain you will need to train it on a corpora of statements from that domain.
 * the website manythings.org provides sentence pairs which can be used as training sets.
 * voice recognition systems use seq2seq networks to turn voice input applitude sample sequences into the thought vector that a seq2seq decoder can turn into a text transcription of the speech. 
*  image captioning - The sequence of image pixels (regardless of image resolution) can be used as an input to the encoder, and a decoder can be trained to generate an appropriate description.

---

summary

* Training a model to generate sequence on input sequences is possible with sequence-to- sequence networks

* Sequence-to-Sequence networks consist out of an encoder and a decoder model

* The encoder model generates a thought vector which is a vector presentation of the input context

* The decoder then uses the thought vector to start predicting output sequences. The entirety of the output sequences form the chat bots response

* Due to the thought vector representation, the input and the output sequence lengths don’t have to match (great for machine translation)

* Thought vectors can only hold a limited amount of information. If longer texts need to be thought vector encoded then Attention is a great tool to encode what is really important



 ---

 * maybe a few more examples of datasets in text.
