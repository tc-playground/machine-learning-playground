# Vocabulary

* Tokenize your text into words and N-grams (tokens
* Build a vector representation of a statement
* Deal with text contractions and abbreviations
* Tokenize social media texts, like tweets from Twitter
* Compress your token vocabulary with stemming and lemmatization • Filter out words with negligible information content (stopwords)
* Handle capitalized words appropriately
* Build a sentiment analyzer from scores for tokens

* Compounds words (ice-cream), Invisible words (Don't!), Implied words (Don't do that!)

* N-grams: pairs, triplets, quadruplets, and even quintuplets of tokens.

* Stemming

* Segementation
  * Breaking a documern into secions: Chapers, paragraph, sentences, words.
  * Tokeinisation: scanner, lexer.
  * Delimitters

* Vocabulary, corpus

### Context-free Grammar
### regular expressions
### finite-statie machines

* Temrinals

* Grammars
  * Fromal
  * Natual Language

• "tokenizer": "scanner", "lexer", "lexical analyzer"
• "vocabulary": "lexicon"
• "parser": "compiler"
• "token", "term", "word", or "n-gram": "token", "symbol", or "terminal symbol"

### Tokenisation

* TOKENIZATION: TURN AN UNSTRCUTURED DOCUMENT INTO A DATA STRUCTUURE (WITHCOUNTS) FOR MACHINE LEARNING
  * cleaning to get consistent spellings for stemming and other steps.

### Ones hot word vectors (matrix) (binary vector) - a data structure for representing a document:
  * Each row in order represents a word in the sentence.
  * Each column represents a particular word in the corpus.
  * The number indicates what word it is (or it's frequency)
  - takes up a lot of space.

### Bag of Words (Vectors) - Document retrival, search.
  * key-word frequency map.

* Pyhton Libs and ADTs
  * dicts, sets, bag of words, binary vector
  * pandas: Series (efficient dict), DataFrames (set of Series)
  * spaCy, NLTK (TreeBank word tokeniser, casual tokenisei, stemmers, WordNetLemmatizer), Stanford CoreNLP.
  * sklearn: VADAR

### Vector Space Model
  * Vector based ops - add, subtract, etc.
  * Dot Product for similarity. (overlapping)

* Contractions
  * rules
  * twitter emoticon tokenisers

### N-grams
  * An n-gram is a sequences containing up to n elements which have been extracted from a sequence of those elements.
    * e.g. sentences, DNA, ...
    * For example, the meaning inverting word "not" will remain attached to it’s neighboring words, where it belongs, rather than "floating free" to be associated with the entire sentence or document.
    * generated by toeknizer.
    * most are ans not usefull for chacterization as either too rare or too common - thse get filtered out.


### Dimensionality reduction

### Stop words
  * Stopwords are common words in any language which occur with a high frequency, but carry much less substantive information about the meaning of a phrase. They often contain relational information though.
    * e.g. a, an, the, this, and, or,  of, on
  * remove them, or have large n-grams?

* vocbulary normalisation
  #### case normalisation - pros and cons
  #### stemming - Stemming removes suffixes from words, in an attempt to combine words with similar meanings together under their common stem. Dimensionality reduction technique. e.g. de-pluralise.
    ##### porter stemmer
    ##### snowball stemmer.
  #### lemmatization - access to information about connections between the meanings of various words we might be able to associate several words together even if their spelling is quite different.
    * e.g. "chat", "chatter", "chatty", "chatting", and perhaps even "chatbot.
    * Accurate lemmatization of a word requires identification of the Part of Speech (POS) of that word because the POS affects its meaning. The POS tag for a word indicates its role in the grammar of a phrase or sentence. e.g. noun.

#### Part of Speech
  * noun, adjective, verb.

### Sentiment Analysis

* Sentiment Analysis
  ##### Heursistic Rule based
  ##### ML based.

* VADAR - rule based sentiment analysis.
   * 7500 keywords in vocabulary.
   * good, neutral, bad - outcomes.

* Naive Bayes - ML based sentiment anyalysis.
   * 


### summary 

* How to implement tokenization and configure a tokenizer for your application
* Several techniques for improving the accuracy of tokenization and minimizing information loss
* N-gram tokenization helps retain some of the "word order" information in a document
* Normalization, stemming and lemmatization consolidate words into groups that improve the "recall" for search engines but reduce precision.
* Stopwords can contain useful information and it’s not always helpful to discard them.

---

### notes

+ nice informal reading style.
+ lots of examples and thought experiments.

* lots of references to futrue chapters.

- mixes in new terminolgy in a wierd order.
- no definitions of things (even informally)
- intorduces python terminology randomly (lambda, list comprehensions, python generators)
- no description at all of naive bayes




