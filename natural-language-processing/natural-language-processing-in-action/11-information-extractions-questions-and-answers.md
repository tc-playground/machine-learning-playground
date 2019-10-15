# Information Extraction and Question Answering

---

* Sentence segmentation
* Named entity recognition
* Numerical information extraction
* POS tagging and dependency tree parsing
* Logical relation extraction and knowledge bases

---

 * we need to extract information or knowledge from natural language text.

---

# Named Entities
 * We’d like our machine to extract pieces of information, facts, from text so it can know a little bit about what a user is saying
 * we’d need to know that "me" represents a particular kind of named entity, a person. 
 * should "expand" or normalize that word by replacing it with the username of the human that made that statement. 
 * should recngnise "google.com" is an abbreviated URL, a named entity
 * spelling of this particular kind of named entity might be "http://www.google.com"
 * recognize that Monday is one of the days of the week (another kind of named entity called an "event") and be able to find it on the calendar.
 * recognize the implied subject of the sentence, "you"
 * A typical sentence may contain several named entities of various types:
    * geographic entities 
    * organizations
    * people
    * political entities
    * times (including dates) 
    * artifacts
    * events
    * natural phenomena

# Relations
 * facts about the relationship between the named entities in the sentence.
 * extract the relation between the named entity "me" and the command "remind".

 # Knowlege base
  * run information extraction on a large corpus, like Wikipedia
  * knowledge base can later be queried make informed decisions or inferences about the world.
  * store knowledge about the current user "session" or conversation.
  * build up information about the current context 
  * Commercial chatbot APIs like IBM’s Watson, or Amazon’s Lex, typically store context separate from the global knowledge base of facts that it uses to support conversations with all the other users.
  * Context can include facts about the user, the chatroom or channel, or the weather and news for that moment in time.
  * Context can even include the changing state of the chatbot itself, based on the conversation. 
  * An example of "self-knowledge" a smart chatbot should keep track of is the history of all the things it has already told someone or the questions it has already asked of the user, so it doesn’t repeat itself.
  * to understand what it reads
  * knowlegebase examples: NELL or Freebase

# knowledge graph
 * Extract knowledge: "In 1983, Stanislav Petrov, a lieutenant colonel of the Soviet Air Defense Forces, saved the world from nuclear war."
   * ('Stanislav Petrov', 'is-a', 'lieutenant colonel')
 * a graph of triplets in the form of (subject, relation, object).
 * A collection of these triplets is a knowledge graph. This is also sometimes called an "ontology".
 * This logical operation of deriving facts from a knowledge graph is called knowledge graph inference or just "inference"
 * a base of knowledge helps a machine understand more about a statement than it could without that knowledge.
 * One of the most daunting challenges in AI research is the challenge of compiling and efficiently querying a knowledge graph of common sense knowledge.
 * There aren’t any common sense knowledge Wikipedia articles for our bot to do information extraction on.
 * some of that knowledge is instinct, hard- coded into our DNA.
 * There are hard-coded common-sense knowledge bases out there for you to build on. Google Scholar is your friend in this knowledge graph search.
 * normalise string or ID to represent a particular relation or noun. 
 * A knowledge base can be used to build a practical type of chatbot called a question answering system (QA system).

# common knowledge 
 * Humans start acquiring much of our common sense knowledge even before we acquire language skill.
   * We don’t spend our childhood writing about how a day begins with light and sleep usually follows sunset.
   * We don’t edit Wikipedia articles about how an empty belly should only be filled with food rather than dirt or rocks.
 * Hard to find a corpus of common knowledge to learn from.


# factual relationships
 * "kind-of", 
 * "is-used-for"
 * "has-a"
 * "is-famous-for"
 * "was-born"
 * "has-profession.
 * NELL, the Carnegie Mellon Never Ending Language Learning bot is focused almost entirely on the task of extracting information about the 'kind-of' relationship.

* Humans are bad at remembering facts accurately, but good at finding connections and patterns between those facts, something machines have yet to master.

# information extraction
 * converting unstructured text into structured information stored in a knowledge base or knowledge graph.
 * Natural Language Understanding
 * Instead of giving our machine fish (facts) we’re teaching it how to fish (extract information)

# regular patterns
 * identify sequences of characters or words that match the pattern so we can "extract" them from a longer string of text
 * tedious to program explicit pattern dectection and brittle.
 * regular expressions - finite state machine of if-then expressions.
 * In computer science and mathematics, the word "grammar" refers to the set of rules that determine whether or a sequence of symbols is a valid member of a language
 * our statistical or data-driven approach to NLP has limits.
 
# Information Extraction
 * Even with machine learning approaches to natural language processing, we need to do feature engineering.
 * Information extraction is just another form of machine learning feature extraction from unstructured natural language data, like creating a bag of words, or doing PCA on that bag of words
 * Information extraction can be accomplished beforehand to populate a knowledge base of facts.
 * Alternatively, the required statements and information can be found on-demand, when chatbot is asked a question or a search engine is queried.

# Grammars and Languages
 * "grammar" refers to the set of rules that determine whether or a sequence of symbols is a valid member of a language, often called a computer language or formal language.
 * a computer language, or formal language, is the set of all possible statements that would match the formal grammar that defines that language.
 * Any formal grammar can be used by a machine in two ways:
   1. to recognize "matches" to that grammar 
   2. to generate a new sequence of symbols
 * A true finite state machine (FSM) can be guaranteed to always run in finite time (to "halt"). 
 * It will always tell us whether we’ve found a match in our string or not. It will never get caught in a perpetual loop (unless you cheat)
 * No "look-back" or "look-ahead" cheats.

# Extracting Named Entities
 * Numbers
 * Dates (temporal)
 * Question Trigger Words 
 * Question Target Words 
 * Named Entities
 * GPS Locations (spatial)

# Extracting Relations
 * extracting knowledge from natural language. 
 * We’d like our bot to learn facts about the world from reading an encyclopedia of knowledge like Wikipedia.
 * This is what most people think of when they hear the term natural language understanding. To understand a statement you need to be able to extract key bits of information and correlate it with related knowledge.
 * knowledge base: The edges of our knowledge graph are the relationships between things. And the nodes of our knowledge graph are the nouns or objects found in our corpus.
 * SUBJECT - VERB - OBJECT.

# POS Tagging
 * POS tagging can be accomplished with language models that contain dictionaries of words with all their possible parts of speech.
 * SUBJECT - VERB - OBJECT. or more complex.
 * They can then be trained on properly tagged sentences to recognize the parts of speech in new sentences with other words from that dictionary. NLTK and SpaCy both implement POS tagging functions.
 * SpaCy parsed sentences also contain the dependency tree in a nested dictionary.
 * This is called semantic drift.


# Entity Name Normalization
 * A normalized representation for entities enables our knowledge base to connect all the different things that happened in the world on that same date to that same node (entity) in our graph.
 * resolving ambiguities is often called "coreference resolution" or "anaphora resolution",
 * similar to lemmatization
 * Need to update a knowledge base after normalisation.

# Realtionship Normalization.
 * Now we need to a way to normalize the relationships, to identify the kind of relationship between entities
 * This will allow us to find all birthday relationships between dates and people, or dates of occurrences of historical events, like the encounter between "Hernando de Soto" and the "Pascagoula people." And we need to write an algorithm to chose the right label for our relationship.
 * these relationships can have a hierarchical name, like "occurred-on/approximately" and "occurred-on/exactly", to allow us to find specific relationships or categories of relationships. 
 * Relationships can also be labeled with a numerical property for the "confidence", probability, weight, or normalized frequency (analogous to TFIDF for terms/words) of that relationship. 

 # Word Patterns
  * Word patterns are just like regular expressions, but for words instead of characters.
  * Instead of character classes we have word classes. 
    * So, for example, instead of matching a lowercase character we might have a word pattern decision to match all the singular nouns ("NN" POS tag).
  * accomplished with machine learning. 
    * Some seed sentences are tagged with some correct relationships (facts) extracted from those sentences. 
    * A POS pattern can be used to find similar sentences where the subject and object words might change or even the relationship words.
  * spaCy
    * PhraseMatcher 
    * Matcher
 * To ensure that the new relations found in new sentences are truly analogous to the original seed (example) relationships, it’s often necessary to constrain the subject, relation, and object word meanings to be similar to those in the seed sentences. 
  * This can best be done with some vector representation of the meaning of words. 
  * Word vectors one of the most widely used word meaning representations used for this purpose.
  * This helps minimize semantic drift.
  * Using semantic vector representations words and phrases has made automatic information extraction accurate enough to build large knowledge bases automatically

# Segmentation

# Sentence Segmentation

# Punctation




 ---

 python: SpaCy











---

NB: How should a system go about aquiring knowledge it needs/may need but does not have? Re-enforcement learning? Thought experiments?

NB: A FSM is like an unrolled Turing Machine - with no loops. Like a specific instance of a turing machine with input defined. FINITE cannot do INFINITE!
    No "look-back" or "look-ahead" cheats. 

NB: Learn GPS locations? A human who does not know them will probably not be able to recognise them. Need to find a definition, or somehow work them out
    by 'relating' them to other things by analogy, then stumbling accorss the earth, then forming a hypothesis and testing it.



