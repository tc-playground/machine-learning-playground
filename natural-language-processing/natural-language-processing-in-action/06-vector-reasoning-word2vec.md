# Word Vectors

---

* Learning what word vectors are
* Understanding the key concepts of word vectors
* Using pretrained models for your applications
* Reasoning with word vectors to solve real problems • Some surprising uses for word embeddings

---

# Word Vectors
  * Word vectors are numerical representations of word semantics, or meaning.
  * Concepts.
  * Compact vector of floating point values that we can do queries and logical reasoning with.
  * Use the context of a word.
  * Create much smaller bags of words from a "neighborhood" of only a few words, typically less than ten tokens.
  * Don't allow these word neighborhoods to span across sentences.
  * Identify synonyms, antonyms.
  * Identify words belonging to the same category: people, animals, places, plants, names, or concepts.
  * Capture all of the literal meanings of a word.
  * Captrue implied or hidden meanings.
  * Word Vectors are biased! Based on the training corpus.

# Semantic Queries and Analogies - word vector maths
  * She invented something to do with physics in Europe in the early 20th century.
    * answer_vector = wv['woman'] + wv['Europe'] + wv[physics'] + wv['scientist']
  * remove biases (e.g. gender bias)
    * answer_vector = wv['woman'] + wv['Europe'] + wv[physics'] + wv['scientist'] - wv['male'] - wv['man'] - wv['man']

# Analogy questions - word vector maths
  * Who is to nuclear physics what Louis Pasteur is to germs?
    * answer_vector = wv['Louis_Pasteur'] - wv['germs'] + wv[physics']

# word2vec
  * train a neural net to predict word occurrences the near each target word.
  * learn the meaning of words merely by processing a large corpus of unlabeled text.
    * All you need is a corpus large enough to mention "Marie Curie" and "Timbers" and "Portland" near other words associated with science or soccer or cities.
  * teach the network to predict words near the target word in our unlabeled sentences.
  * We don’t care about the accuracy of those predictions, we just care about the internal representation, the vector, that word2vec gradually builds to help it generate those prediction.
  * word2vec will learn about things you might not think to associate with all words. Did you know that every word has some geography, sentiment (positivity), and gender associated with it?
  * Companies that deal with large corpora and can afford the computation have open sourced their pretrained word vector models.

# unsupervised learning
  * learn clusters from unlabelled data.
  * e.g. text.

# LSA Topic Vectors
  * Topic vectors constructed from entire documents using LSA are great for document classification, semantic search, and clustering.
  * Aren’t accurate enough to be used for semantic reasoning or classification and clustering of short phrases or compound word.

# vector-oriented reasoning
  * add and subtract word vectors to reason about the words they represent and answer question
  * train the single-layer neural networks required to compute these new word vectors.
  * vector vectors of words allowed the same arithmetic that you might have seen in your calculus or linear algebra
  * the word2vec model contains information about the relationships between words including similarity.
  * Word2vec allows us to transform our natural language vectors of token occurrence counts and frequencies into the vector space of much lower-dimensional Word2vec vectors
  * In this lower dimensional space we can do our math, and then convert back to a natural language space
  * the difference between a singular and a plural word is often roughly the same and in the same direction
  * text generation and text classification

# word vectors 2
 * represent the semantic meaning of words as vectors in the context of the training corpus.
 * two possible ways to train word2vec embeddings:
   * The skip-gram approach predicts the context of words (output words) from a word of interest (the input word
   *  The Continuous Bag of Words (CBOW) approach predicts the target word (the output word) from the nearby words (input words).

# skip-gram
 * The skip-gram approach predicts the context of words (output words) from a word of interest (the input word).
 * Skipgrams are n-grams which contain gaps due to the fact that we skipped over tokens.
 * In the skip-gram training approach, we are trying to predict the surronding window of words based on an input word.
   * e.g. Claude Monet painted the Grand Canal of Venice in 1908. 
     * target word       (input)           : painted
     * surrounding words (predicted ouput) : 'Claude Monet' and 'the Grand'
 * the network 
   * consists of two layers, 
   * the hidden layer consists of N neurons where N is the number of vector dimensions used to represent a word. 
   * Both the input and output layers contain M neurons, where M is the number of words in the vocabulary of the model. 
   * The output layer activation function is a softmax,
 * The softmax activation of the output layer nodes (one for each token in the skip-gram) calculates the probablity of a output word being found as a surrounding word of the input word.
 * The output vector of word probabilities can then be converted into a one-hot vector where the word with the highest probablity will be converted to 1 and all remainind terms will be set to 0.
 * Thanks to the one-hot vector conversion of our tokens, each row in the weight matrix is representing a different word from your corpus.
 * After the training is complete and you decide not to train your word model any further, the output layer of the network can be ignored. 
 * Only the weights of the inputs to the hidden layer are used as the embeddings.
 * The weight matrix is your word embedding. 
 * The dot product between the one-hot vector representing the input term and the weights then represents the word vector embedding.
 * skip-gram approach works very well with small corpora and rare terms. This is because with the skip gram approach you’ll have more examples due to the network structure.

# continuous bag of words
 *  The Continuous Bag of Words (CBOW) approach predicts the target word (the output word) from the nearby words (input words).
 * In the Continous Bag of Words (CBOW) approach, we are trying to predict the center word based on the surrounding words.
 * flipped version of skipgram
    * e.g. Claude Monet painted the Grand Canal of Venice in 1908. 
     * surrounding words (input)            : 'Claude Monet' and 'the Grand'
     * target word       (predicted output) : painted
 * A sliding window of words.


# Frequent Bi-Grams - word2vec enhancement
 * "Elvis Presley" -> Probability of Elvis given Presley is high. Can use a custom bigram/trigram vocab dictionary.

# Subsampling frequent tokens - word2vec enhancement
 * Common words like the or a often don’t carry significant information
 * To avoid this, frequent words are sampled less often during training to not overweight them in the word2vec vector space.
 * They are given less influence than the rarer, more important words in determining the ultimate embedded representations of word
 * for example:
   * So, if a word shows up 10 times across your entire corpus of 1 million words, with a subsampling threshold of 10^{-6}, then the probability of keeping the word in any particular N-gram is 68%. 
   * You would skip it 32% of the time while composing your N-grams during tokenization.

# Negative Subsampling
 * Instead of updating all weights of the words which weren’t included in the word window, the Google team suggested to sample a few negative samples to update their weights.
 * Instead of updating all weights, we pick n negative samples (words which didn’t match our expected output) and their their weights.
 * The computation can be reduced dramatically and the performance of the trained network doesn’t decrease significantly.

# gensim - generate your word2vec word-vectors
 * break our documents into sentences and the sentences into tokens.
 * train domain-specific word2vec model
 * you can freeze the model and discard the output vector to save space (5o%), but, then you cannot continue training.

# Word2vec vs GloVe (Global Vector)
 * backpropagation, which is usually less efficient than direct optimization of a cost function using gradient descen
 * understand the reason why Word2vec worked so well and find the cost function that was being optimized.
 * counting word co-occurrences and recording them in a square matrix they could then compute the singular value decomposition (SVD) of this co-occurrence matrix they could split this co-occurrence matrix into the same two weight matrices that Word2vec produces (if they are normalized the same way and the Word2vec model also converges to the global minimum).
 * GloVe can produce matrices equivalent to the input weight matrix and output weight matrix of Word2vec for a language model with the same accuracy as Word2vec, but in much less time.
 * it uses the data more efficiently so it can be trained on smaller corpora and still converge.
 
# Advantages of GloVe:
 * Faster training
 * Better RAM/CPU efficiency (large corpora)
 * More efficient use of data (small corpora)

# fastText
 * fastText, predicts the next word not only on the previous words as we have seen in Word2vec. 
 * extended the training input to the character n-grams of each word.
 * e.g.  word whisper would generate the following 2 and 3-grams: wh, whi, hi, his, ...
 * The advantage of this approach is that it handle rare words much better than the original Word2vec approach

# Word2vec vs LSA
 * LSA topic-document vectors are the sum of the topic-word vectors for all the words in those documents.
 * If we wanted to get a word vector for an entire document that is analagous to topic-document vectors we’d sum all the word2vec word vectors in each document.
 * If your LSA matrix of topic vectors is of size N_{words} \times N_{topics}, then the LSA word vectors are the rows of that LSA matrix. 
 * These row vectors capture the meaning of words in a sequence of around 200-300 real values like Word2vec does.
 * LSA topic-word vectors are just as useful as word2vec vectors for finding both related and unrelated terms
 * However, word2vec gets more use out of the same number of words in it’s documents by creating a sliding window that overlaps from one document to the next.
 * Adding completely new words would change the total size of your vocabulary and therefore your one-hot vectors would change. :(
 * LSA trains faster compared to Word2vec.
 * For long documents LSA does a better job of discriminating and clustering those documents.
 * word2vec has semantic reasoning.

# Advantages of LSA:
 * Faster training
 * Better discrimination between longer documents

# Advantages of Word2vec:
 * Scales better to larger vocabularies and corpora • Supports semantic reasoning including analogies

# Visualisation
 * PCA (dimensions 2, 3) then plot with grpahing tool.

# Unnatural Words 
 * Word2vec are useful not only for English words but also for any sequence of symbols where the sequence and proximity of symbols is representative of their meanin
 * other languages and alphabets
 * ciphers

# Doc2vec
 * The concpets of word2vec can be applied to documents.
 * 


----

# soft-max
  * The softmax function is often used as the activation function in the output layer of neural networks when the network’s goal is to learn classification problems.
  * The softmax will squash the output results between 0 and 1 and the sum of all output notes will always add up to 1. 
  * That way, the results of an output layer with a softmax function can be considered as probabilities.

---

Reources

* pre-trained word2vec models

1.  https://github.com/3Top/word2vec-api#where-to- get-a-pretrained-model
2.  https://bit.ly/GoogleNews-vectors-negative300
3.  https://github.com/facebookresearch/fastText


* python -  Gensim’s word2vec
* Detector Morse - python sentence sgmentation.

# GloVe

# FastText.

---

Notes

+ advtanges of LDA vs word2vecv vs Glove

* Missing figure 3?
* some sort of ongoing example for all chapters?







