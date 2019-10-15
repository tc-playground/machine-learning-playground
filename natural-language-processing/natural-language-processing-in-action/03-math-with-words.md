
### Introduction

* Counting words and Term Frequencies to analyze meaning
* Predicting word occurrence probabilities with Zipf’s Law
* Vector representation of words and how to start using them
* Finding relevant documents from a corpus using Inverse Document Frequencies
* Estimating the similarity of pairs of documents with Cosine Similarity and Okapi BM25

---

#### which words are more important to a particular document and across the corpus as a whole.

#### Uncanny valley: A region where a machine starts to appear lifelike, but retains its machine awkwardness.

#### Shallow Learning

#### NLP Representations
* Bags of Words: vectors of word counts or frequencies
* Bags of N-Grams: counts of word pairs, triplets, etc (we’ll get to this in a chapter or two) 
* TF-IDF Vectors: word scores that better represent their importance

#### Bag of Words (TF Vector)
* Words as keys; frequencies (from document) or other stats as value.
* Term Frequency (TF) - The number of times, the word, n-gram, etc appears in the document, divided by the number of terms in the document.

### Vectorising
* Each document is represented by an ordered array of terms from the lexicon associated with their TF count. (NB: Non-sparse. Values are ommited if not present.)

* Lexicon: The collections of words in our vocabulary. 
* Corpus:  The collection of documents for which the lexicon is defined.

### dot product of two of our vectors — multiply the elements of each vector pairwise and then sum those products up.
### norm (or mod) of vector — the square root of the sum of the squares of its elements.

### Vector Space
* NLP used euvlidean n-dimensional vector spaces.
* Dimensionality (K): Number of distinct word in the corpus.
* The vector of the document defines a point in N-D space.
* Similarity:
    * Two vectors are "similar" if they are closetogether. (poor measure)
    * Two vectors are "similar" if they share similar direction.

### Ccurse of Dimensionality : Vectors will get exponentially farther and farther away from one another, in Euclidean distance, as the dimensionality increases.
* A lot of simple operations become impractical above 10 or 20 dimensions

### 2-Norm Distance - Te Euclidean distance between the vectors by subtracting them and computing the length of that distance between them. As the crow flies.

### Cosine Similarity (dot-product)
* A dot B = mod(A) * mod(B) * cos(theta)
* cos(theta) = A dot B / (mod(A) * mod(B))
* Range between -1.0 and 1.0.
* cosine similarity of 1 -the 'same' vector. The documents are using similar words in similar proportion.
* cosine similarity of 0 -the 'orthogonal' vector. The documents are orthogonal and share no components.
* cosine similarity of -1 -the 'opposite' vector. Cannot happen for simple word counts- as counts of words can never be negative. So word count (term frequency) vectors will always be in the same "quadrant" of the vector space. NB: Opposite concepts might be describe this way though.

### Zipfs Law - Given some corpus of natural language utterances, the frequency of any word is inversely proportional to its rank in the frequency table
* For example, in a *ranked* list of words frequesncies in a document - the first item will appear twice as much as the second, and three times as much as third...
~ for a sufficiently large sample, the first word in that ranked list is twice as likely to occur in the corpus as the second word in the list is. And it is four times as likely to appear as the fourth word on the list. 
~ So given a large corpus, we can use this break down to say statistically how likely a given word is to appear in any given document of that corpus.


### Topic Modelling

#### Term Frequency - The number of times a term exists in a document.

#### Inverse Document Frequency (IDF) - The IDF of a term is the ratio of the total number of documents to the number of documents the term appears in.
  * how strange is it that this token in this document (ert the corpus)?
  * usually taken as a logarithm so large corpuses have less variance.
  * log(total_docs_in_corpus/num_docs_in_corpus_with_term)
  * There a re lots of different ways to normalise the TF with different IDF functions: t-tests, chi-squared, ATC, LTU, etc.


#### TF-IDF - for corpus D, document d, and term t => a measure of likely the specified document is to be a 'about' the term. A MEASURE OF RELEVANCE.
  * TF * IDF => frequency_of_term_t_in_doc_d * log(total_docs_in_corpus_D/num_docs_in_corpus_D_with_term_t)
  * So the more times a word appears in the document the TF (and hence the TF-IDF) will go up. 
  * At the same time, as the number of documents that contain that word goes up, the IDF (and hence the TF- IDF) for that word will go down.
  * given (t, d, D) TF-IDF is a a measure of 'relevance' d is about t.


# Relevance Ranking - TF-IDF based search
  * Index: For each document d, generate the 'document term vector': {Term_t -> IDF-TF(D, d, t)} : dn
  * Query: by creating a 'search term vector': s1
  * Search: For each document calculate the Cosine Similarity of s1 with each dn : cs_n
  * Order the results by cs_n.

# Inverted Index

# Laplace SMoothing / Additive Smoothing - Add 1 to 0 TF to prevent division by 0.

* Chatbot - Instead of returning similar TF_IDF documents, return an 'answer' associated with the document.


---

* python
  * whoosh - inverted index/ search engine
  * scipy
  * sklearn - TFIDFVectorizer model is a sparse numpy matrix,


---

### Notes

- Slightly confusing details when defining TF.
- bits are unclear / ambiguous.
- starts to redfine some things that seems similar to chapter 01, but, with different terminology.
- jump between very simplistic 2-D vector space.
- unecessary details.
- could be simpler - given (t, d, D) TF-IDF is a a measure of 'relevance' d is about t.

* zipfs law - could plot Brown example?
* inverted index would have been cool.

+ good description of the meaning of cosine similarity.


