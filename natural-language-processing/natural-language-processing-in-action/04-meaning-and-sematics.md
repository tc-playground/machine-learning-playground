### Introduction

* Creating topic vectors from text
* Understanding Latent Semantic Analysis (LSA)
* Creating topic vectors words in your lexicon
* Estimating the semantic similarity between documents or words • Implementing semantic search
* Scaling semantic search for large corpora
* Using semantic topics as features in your NLP pipeline
* Navigating high-dimensional vector spaces

---

* extract the meaning (semantics) from a text and it’s TF-IDF vector representation.

#### Latent Semantic Analysis (LSA) or Latent Semantic Indexing
     * Represent the meaning of entire documents with those vectors (topic vectors). 
     * The vectors it produces can be as compact or as expansive as you like. LSA topic-document and topic-word vectors can have as few as 1 dimension or as many as 1000’s of dimensions.

#### Word Vectors

#### Topic Vectors - Enable Vector Oriented Learning.
  ##### Topic Document Vectors - 1 for each document in the corpus.
  ##### Topic Word Vectors - 1 for each term (words, stems, lemma, n-grams) in the Lexicon.

* the relationships between words and topics is symmetric - transpose topics into word vectors.

```
topic['pet']     = (.3 * tfidf['cat']  + .3 * tfidf['dog'] +  0 * tfidf['apple'] +  0 * tfidf['lion'] - .2 * tfidf['NYC'] + .2 * tfidf['love'])
topic['animal']  = (.1 * tfidf['cat']  + .1 * tfidf['dog'] - .1 * tfidf['apple'] + .5 * tfidf['lion'] + .1 * tfidf['NYC'] - .1 * tfidf['love'])
topic['city']    = ( 0 * tfidf['cat']  - .1 * tfidf['dog'] + .2 * tfidf['apple'] - .1 * tfidf['lion'] + .5 * tfidf['NYC'] + .1 * tfidf['love'])

VVV - Custom LSA compresses 6 diemnsions into 3 dimensions...

word_vector['cat']   =  .3 * topic['pet'] + .1 * topic['animal'] +  0 * topic['city']
word_vector['dog']   =  .3 * topic['pet'] + .1 * topic['animal'] - .1 * topic['city']
word_vector['apple'] =   0 * topic['pet'] - .1 * topic['animal'] + .2 * topic['city']
word_vector['lion']  =   0 * topic['pet'] + .5 * topic['animal'] - .1 * topic['city']
word_vector['NYC']   = -.2 * topic['pet'] + .1 * topic['animal'] + .5 * topic['city']
word_vector['love']  =  .2 * topic['pet'] - .1 * topic['animal'] + .1 * topic['city']
```

# Polysemy - the existence of words and phrases with more than one meaning.
## Homonyms - words with the same spelling and pronounciation, but different meanings.
## Homographs - words spelt the same, but with different pronounications and meanings.
## Zeugma - use of two meanings of a word simultaneously in the same sentence.
## Homophones - words with the same pronounication, but different spellings and meanings (an NLP challenge when generated text will be spoken)


* find those word dimensions that "belong" together in a topic and add their TF-IDF values together to create a new number to represent the amount of that topic in a document. 
* weight them for how important they are to the topic
* have negative weights for words that reduce the likelihood that the text is about that topic

# L1 Norm - Manhatten/block/taxi distance (distance counted by dimensional movements) : (absolute value of our word weights sum to 1.0 for each topic.)
# L2 Norm - Euclidean distance ((pythagorean)) : (LSA uses this measure)

* LSA - "transform" a vector from one vector space (TF-IDFs) to another (topic vectors). 
* So we want an algorithm that can create a matrix of n terms by m topics that we could multiply (inner product) with a vector representing the words in a document to get our new topic vector for that documen

* Topic Vector Generations Algorithmss - Each of these techniques can be used to create vectors to represent the "meaning" of a word or collection of words (a document).
 # Latent Dirichlet Allocation (abbreviated LDiA to distinguish it from LDA, below)
 # Latent Semantic Analysis/Indexing (LSA or LSI) or Principal Component Analysis (PCA) • Linear Discriminant Analysis (LDA)
 # Quadratic Discriminant Analysis (QDA)
 # Random Projection (RP)
 # Nonnegative Matrix Factorization


#  "Singular Value Decomposition" (SVD) - Linear Algebra for LSA
  * LSA uses SVD to find the combinations of words that are responsible, together, for the biggest variation in the data.
  * We can rotate our TF-IDF vectors so that the new dimensions (basis vectors) of our rotated vectors all align with these maximum variance directions.
  * The "basis vectors" are the axes of our new vector space and are analogous to our topic vectors.
  * Each of our dimensions (axes) becomes a combination of word frequencies rather than a single word frequency. 
  * So you we think of them as the weighted combinations of words that make up various "topics" in our corpus.
  * The machine doesn’t really "understand" what the combinations of words means, just that they go together. A human might look at them and lanbel them though.
  * They may also contain opposite word: cat, feral, domesticated.
  * Dimesionality reduction - discard dimensions (topics) that have the least variance.
  * LSA compresses more meaning into fewer dimensions. 

# "Principal Component Analysis" (PCA)
  * PCA to images; is what SVD is to text.

# Context MAtrix
  * Both the BOW term-document matrix and the TF-IDF term-document matrix are often called the "context matrix." These matrices provide the context in which a word was found, "the company that a word keeps."

# SVD
  * SVD is an algorithm for decomposing a matrix into three "factors", three matrices that can be multiplied together to recreate the original matrix.
  * This is analogous to finding the integer factors that can be multiplied to "recreate" the integer that you factored, only we need exactly three factors for our matrix of real numbers.
    * But our factors aren’t scalar integers, they are 2-D matrices.
    * And there can be any number of integer factors and we are looking for only 3 very specific matrix factors.
    * Our matrix factors contain real numbers (floats) rather than integers.
  *  We run this SVD algorithm on a __term-document matrix__, on the normalized __TF-IDF term-document matrix__.
  * SVD will help us find combinations of words that belong together, because they occur together a lot
  * SVD finds those co-occurring words by calculating the __correlation between the columns (terms)__ of our term-document matrix, which is just the square root of the dot product of two columns (term-document occurrence vectors).

  * SVD will group those terms together that have high correlation (because they occur in the same documents together a lot). And we can then think of these collections of words as "topics" and this will help us turn our bags of words (or TFIDF vectors) into topic vectors that tell us the topics a document is about. A topic vector is kind-of like a summary, or generalization, of what the document is about, or at least what the Bag of Words is about.

  * The semantic similarity between two natural language expressions (or individual words) is proportional to the similarity between the contexts in which words or expressions are used (co-occurrence correlation).

  # Word Co-occurance - LSA
   * The usefulness of word cooccurrence. Can you figure out what "awas" means from its context in this statement? Awas! Awas! Tom is behind you! Run!
   * Can you see how shorter documents, like sentences, are better for this than large documents like articles or even entire books? This is because the meaning of a word is usually closely related to the meanings of the words in the sentence that contains it. But this isn’t so true about the words at the beginning and end of a longer document.

* NB: In machine learning document vectors are normally horizontal; in maths thyes are normally vertical - so need to be transposed.

# SVD Formula: Wmn = Ump * Spp * VTmn

# m              : number of terms in vocbulary.
# n              : number of documents in corupu.
# p              : number of topics in corupus.
# W              : Document-Term Matrix (Document TD-IDF matrix) [x: documents, y: terms]
# U              : Term-Topic Matrix (LSA Corpus) [x: terms, y: topic]
# S              : Topic-Topic Eigenvalue Matrix (Topic-Covariance) (Diagonal Matrix) [x: topic, y: topic]
# P (transposed) : Topic-Document Matrix (LSA Compat) [x: document, y: topic]

# Matrix Factor U
  * Contains all the topic vectors for each word in our corpus as columns.
  * Is a transformation that can convert a TF-IDF vector into a topic vector.
  * We can just multiply our topic-word U matrix by any word-document column vector to get a new topic- document vector.
  * This is because the weights or scores in each cell of the U matrix represent how important each word is to each topic.

# Matrix Factor S
  * Diagonal matrix in contaiing the eigenvalues or singular values of our decomposed matri.
  * The variance of each topic.
  * The "explained variance" that each of the topics contributes to the spread of the document TF-IDF vectors around in the vector space of term frequencies.
  * A larger value says that the topic associated with that column or row is really important to explaining the choice of words for a given document in our corpus. That topic is used in a lot of documents distributed throughout our vector space.
  * To reduce dimensions without reducing information and "meaning" in our topic vectors we better hang on to the topics that have large eigenvalues.
  * We can ignore the S matrix as well, once we’ve used it to determine the most important topics in our topic vector.
  * S represents the "scaling" part of the linear transformation of a vector, we’re only interested in the "direction:.

# Matrix factor V.
  * All the topic vectors for each document as columns (before it’s transposed to make the inner product work).
  * This is the thing we want to be able to compute from a set of TF-IDF vectors, like the W matrix for our corpus on the far left.
  * This is the "answer", the topic vector for each document in our corpus, but we also want to be able to compute it on a new bag of words or TF-IDF vector.
  * So we don’t need to record this matrix. Instead we can compute it anew whenever we need it.
  * To create a row in this matrix all we need to do is multiply the inverse of the V matrix (or transpose, in our case) by any new TF-IDF vector to get the normalized topic vector for the document in that vector.


# Consine Distance Similarity.
  * Cosine distance only cares about the angle between things.
  * We normalized the length of all our TF-IDF vectors to be one, unit length (2-norm of 1).
  * Unit length vectors make machine learning a lot easier. If you don’t have to worry about scale, it’s one less parameter your model has to try to guess about from the data.

 # 2-Norm 1 vector normalisation.

 # Truncated SVD
  * Machine Learning Model. 

 # Principal Component Analysis (PCA)
   * Like SVD/LSA - often used in image processing.
   * LSA (SVD) preserves the structure, information content, of our vectors by maximizing the variance along the dimensions of our lower dimensional "shaddow" of the high- dimensional space.
   * This is what we need for machine learning so that each low dimensional vector captures the "essence" of whatever it represents.
   * LSA (SVD) maximizes the variance along each axis. And variance turns out to be a pretty good indicator or "information" or that "essence".

# Over-fitting
  * More workds in lexicon than documents in corpus. => clustering / dimension reduction/consolidation.
  * Exactly what LSA is for.
  * LSA will reduce our dimensions and thus reduce overfitting.
  * 9232-D TF-IDF vectors into 16-D topic vectors.

# Truncated SVD
  * with large datasets you’ll want to use TruncatedSVD instead of PCA.
  * transform any TF-IDF vector into a topic vector (with less dimensions: 9232 -> 16).
  * many iterations, centering vlaues around 0.
  * One way to find out how well a vector space model will work for classification is to see how will cosine similarities between vectors correlate with membership in the same class. Let’s see if the cosine similarity between corresponding pairs of documents is useful for our particular binary classification.
  * Normalizing each topic vector by it’s length (L2-norm) simplifies the cosine similarity computation into the dot product.
  * this is how semantic search works as well. You can use the distances between a query vector and all the topic vectors for your database of documents to approximate the sematic similarity.
  * When using TruncatedSVD, you should discard the eigenvalues before computing the topic vectors.
    1. Normalizing our TF-IDF vectors by their length (L2-norm)
    2. Centering the TF-IDF term frequencies by subtracting the mean frequency for each term (word)
  * Normalizing eliminates any "scaling" or bias in the eigenvalues and focuses your SVD on the rotation part of the transformation of your TF-IDF vectors.
  * If you want to use this trick within your own SVD implementation you can normalize all the TF-IDF vectors by the L2-norm before computing the SVD or TruncatedSVD.

# LDA - Linear Discriminant Analysis (like k-nearest neighbour?)
  * It just computes the the average position of all the vectors in each class (like our SMS messages labeled spam and nonspam). 
  * These two locations are called the centroids of those point clouds for each class.
  * Classification based on closeness to centroid.
  * the power of LSA (PCA). You can compute topic vectors without any labels on your data. And then they help you extrapolate from a smaller number of labeled examples to all the others in your data set.


# Latent Dirichlet Allocation (LDiA) - uses background knowledge (prior probabilities)
  * unlike LSA, LDiA assumes a Dirichlet distribution of word frequencies.
  * More carefully assigns words in documents into the topic vector - so is easier ti understand.
  * LDiA assumes that each document is a mixture (linear combination) of some arbitrary number of topics, that you select when you begin training the LDiA model.
  * LDiA also assumes that each topic can be represented by a distribution of words (term frequencies).
  * The probability or weight for each of these topics within a document as well as the probability of a word being assigned to a topic is assumed to start with a Dirichlet probability distribution (the prior if you remember your statistics).
  * Generative - How a machine that could do nothing more than roll dice (generate random numbers) could write the documents in a corpus that we want to analyze.
  * Distributiions
    1. Number of words to generate for the document (Poisson distribution)
    2. Number of topics to mix together for the document (Dirichlet distribution).
  * Filter stop words.
  * This LDiA algorithm relies on the BOW vector space model (VSM) of natural language text, like all the other algorithms in this chapter.
  * Generate the topic vectors from data. (must give it a K for number of topics).
  * Find K based on some cost function.
  * The topics produced by LDiA tend to be more understandable and "explainable" to humans. (Where LSA (PCA) tries to keep things spread apart that were spread apart to start with, LDiA tries to keep things close together that started out close together.)
  * 


# Variable co-linearity
  * "Variables are col-inear." This can happen with a small corpus when using LDiA because our topic vectors have a lot of zeros in them and some of our messages could be reproduced as a linear combination of the other messages topics.
  * A problem with the underlying data.
    * Add "noise" or meta data to your SMS messages as synthetic words, or you need to delete those duplicate word vectors. 
    * If you have duplicate word vectors or word pairings that repeat a lot in your documents then no amount of topics is going to fix that.
  * When one word occurs, another word (pair) alwats occurs in the same message.
  * So the resulting LDiA model had to arbitrarily split the weights among these equivalent term frequencies.
  * You can iterate through all the pairings of the bags of words to look for identical vectors. These will definitely cause a "collinearity" warning in either LDiA or LSA.
  * If we ever need to, we can turn down the LDiA n_components to "fix" this. This would tend to combine those topics together that are a linear combination of each other (collinear).

* Topic Moeddling: helps us generalize our models from a small training set so it still works well on messages using different combinations of words (but similar topics).

# LSA
 * We should see that LSA preserves large distances, but does not always preserve close distances (the fine "structure" of the relationships between our documents). 
 * This is because the underlying SVD algorithm is focused on maximizing the variance between all our documents in the new topic vector space. 


# feature vectors:
  * word vectors
  * topic vectors
  * document context vectors
  * etc

# Distance measures
 * Euclidean, Cartesian, root mean square error (RMSE): 2-norm or L2 • Squared Euclidean, sum of squared distance (SSD): L22
 * Cosine (angular, projection): normalized dot product
 * Minkowski: p-norm or Lp 31
 * Fractional, fractional norm: p-norm or Lp for 0 < p < 1
 * City block, Manhattan, taxicab, sum of absolute distance (SAD): 1-norm or L1 • Jaccard, inverse set similarity,
 * Mahalanobis
 * Levenshtein (Edit distance)

# Similarity Scores
  * Similarity scores are designed to range between 0 and 1
  * Distance measures are often computed from similarity measures (scores) and vice versa such that distances are inversely proportional to similarity scores.
  * not all distances qualify to be called metricsL
    1. nonnegativity: metrics can never be negative
    2. indiscerniblity: if a metric between two objects is zero then they must be identical
    3. symmetry: the metric from A to B equals the metric from B to A
    4. triangle inequality: the metric A to C is no larger than the metrics A to B plus B to C


# Steering or "learned distance metrics" (adding meta data) 
 * [20: http://users.cecs.anu.edu.au/~sgould/papers/eccv14- spgraph.pdf]
 * All of the previous approaches to LSA failed to take into account information about the similarity between documents. The feature (topic) extraction models didn’t have any data about how "close" the topic vectors should be to each other. 
 * Are the latest advancement in dimension reduction and feature extraction.
 * By adjusting the distance scores reported to clustering and embedding algorithms, it’s possible to "steer" the your vectors so that they minimize some cost function.
 * In this way you can force your vectors to focus on some aspect of the information content that you are interested in.
 * Only then can the algorithm compute the optimal transformation from your high dimensional space to the lower dimensional space.


# LDA (again)
 * LDA works similarly to LSA, except it requires classification labels or other scores to be able to find the best linear combination of the dimensions in high dimensional space (the terms in a BOW or TFIDF vector). 
 * Rather than maximizing the separation (variance) between all vectors in the new space, LDA maximizes the distance between the centroids of the vectors within each class.
 * Unfortunately, this means you have to tell the LDA algorithm what "topics" you’d like to model by giving it examples (labeled vectors.
 ? The resulting lower-dimensional vector can’t have any more dimensions that the number of class labels or scores that you are able to provide
 * penalize our score more for mislabeling spam messages (false negatives) than for getting normal messages wrong (false positives).
 * LSA + LDA

# Topic Vector Power
  * With topic vectors we can do things like compare the meaning of words, documents, statements and corpora.
  * We can find "clusters" of similar documents and statements. 
  * We’re no longer comparing the distance between documents based merely on their word usage. 
  * We’re no longer limitted to keyword search and relevance ranking based entirely on word choice or vocabulary. 
  * We can now find documents that are actually relevant to our query, not just a good match for the word statistics themselves.

# Semantic Search
  * To find semantic matches, we’d need to search through our entire database of topic vectors for the best match.

# Locality Sensitive Hashing (LSH)
  * LSH usually allows for similar vectors to have similar hashes.
  * These locality-sensitive hashes are like Zip Codes that define more and more precise locations the more digits we add to the Zip Code.

# Imbalanced Datasets
  * Need Balance.


* Topic Vector creation -> Model learning.

* tune this concept of topic vectors so that the vectors associated with words are more precise and useful.


---

Python

* sci-kit (sk learn)
* TruncatedSVD
* sklearn.decomposition.PCA
* gensim.models.LsiModelL runs on batches of documents and merges the results from these batches only at the very end
* Sciki-Learn’s version of LSA (called PCA)
* PCA in Scikit-Learn, this is really just LSA (truncated SVD)
* TfidfVectorizer - sparse matricies.
* built-in python product() : cross-product.
* sklearn.metrics.pairwise : distance metricsa.
* Semantic MAtches: Facebook’s FAISS package and Spotify’s Annoy package. 


---

Notes

+ Good intutition for LSA topic vectors.
+ Good comparisson between BOW vectors and TF-IDF vectors.
+ Good inutition of word occurrance.
+ 3D -> 2D Horse example of PCA/LAS/TruncatedSVD
+ Good tips on SVD normalisation.
+ Colinearity examples.

* LDA sort of discussed twice.
* wouldbe nice if there were clear definition to distinguisg LDA vs LSA vs SVD etc.

- no traditional explanation of standard ML stuff - cross validation etc.
- Not defining L1 and L2 norms clearly.
- inner product not introduced.
- dicussion on 'similarity, measure', etc. is interesting, but, does not add much.
- paragraph headings a bit wierd and mis-ordered.
- Figure 8 does not make much sesne atm.
- summary wrong? How to use semantic search to as part of a dialog engine (chatbot)


---

NB: Is not (not vs opposites vs complements) always equal? Imagine not as a negative weight in an NLP process. There are an infinte number of negative things we could add. Only the true opposties would have a large value to denote it's presence detracting from a topic. Hmmm... ROC Analysis.

NB: PCA produces a lower dimensional 'shadow' of the original object. Quite like Pato's cave!

NB: Associating topic vectors with words/labels?

NB: Classifier for Plato vs Aristotle quotes.

NB: Context: "LSA preserves large distances, but does not always preserve close distances (the fine "structure" of the relationships between our documents)". Spatially, a context must always be a larger entity than the subjects. Context, encompassing large space, large structures; subjects, internal spaces, fine grained. How do things get their semantics from their context?









