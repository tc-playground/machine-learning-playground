# Getting Chatty

---

* Four ways to build a smarter chatbot
* What AIML is all about
* What makes a Chatbot pipeline different from other NLP pipelines
* A hybrid chatbot architecture that combines the best ideas into one
* How to use machine learning to make your chatbot get smarter and smarter over time • "Agency", simulating spontaneous statements with a chatbot

---

# ELIZA - first chatbot

# NLP chatbot techniques
 * Tokenization, stemming and lemmatization
 * Vector space language models like bags of words vectors and topic vectors (LSA) • Nonlinear language representations like word2vec or LSTM "thought vectors"
 * Word sequence-to-sequence translators
 * Pattern matching and templates

# chatbots - historical evolution
 1. Grammar: pattern matching and response templates ("canned" responses) 
 2. Grounding: logical knowledge graphs and inference on those graphs (context)
 3. Search: text retrieval
 4. Generative: statistics and machine learning
 * The most advanced chatbots use a hybrid approach that combines all of these techniques.
 * The four basic chatbot approaches can be combined in a variety of ways to produce useful chatbots.

# chatbot application
 * Question Answering: Google Search, Alexa, Siri
 * Virtual Assistants: Google Assistant, Alexa, Siri, MS paperclip
 * Conversational: Google Assistant, Google Smart Reply, Mitsuki Bot
 * Marketing: Twitter bots, blogger bots, Facebook Messenger bots, Alexa, Allo • **Customer Service: Storefront bots, technical support bots
 * Community Management: Bonusly, Slackbot
 * Therapy: Wysa, YourDost, Siri, Allo

# Question Answering Systems
 * Most question-answering systems search a knowledge base first to "ground" them in the real world. 
 * If they can’t find an acceptable answer there, then they may search a corpus of unstructured data (or even the entire Web) to find answers to your question.

# Virtual Assistants
 * helpful when you have a goal in mind
 * Goals or intents are usually simple things like launching an app, setting a reminder, playing some music, or turning on the lights in your home. For this reason.
 * Lawyers are certainly goal-based virtual assistants!
 * "voice first" design, when your app is designed from the ground up around a dialog system

# Conversational Chatbots
 * In a typical Turing Test, humans interact with another chat participant through a terminal and try to figure out if it is a bot or a human. 
 * Mitsuku: http://www.square-bear.co.uk/aiml/ 

# Marketing Chatbots
 * Marketing chatbots are designed to inform users about a product and entice them into purchasing it. 
 * Some virtual assistants are actually marketing bots in disguise. Consider Amazon Alexa and Google Assistant, though they claim to assist you with things like adding reminders and searching the web, they invariably prioritize responses about products or businesses over responses with generic or free information. These companies are in the business of selling stuff, directly in the case of Amazon, indirectly in the case of Google.
 * Most marketing chatbots are conversational, to entertain users and mask their ulterior motives. 
 * They can also employ question answering skills, grounded in a knowledge base, in order to inform your about products or the companies behind them.

# Community Management Chatbots
 * Community management is a particularly important application of chatbots because it influences how society evolves. 
 * A good chatbot "shepherd" can steer a video game community away from chaos and help it grow into an inclusive, cooperative world where everyone has fun, not just the bullies and trolls. 
 * A bad chatbot, like the twitter bot Tay, can quickly create an environment of prejudice and ignorance.
 * Chatbots seem to do more than merely reflect and amplify the best and the worst of us. 
 * They are an active force, partially under the influence of their developers, trainers, for either good or evil.

# Customer Service Chatbots
 * Customer service chatbots are often the only "person" available when you visit an online store. 
   * IBM’s Watson 
   * Amazon’s Lex
 * They often combine both question answering skills (remember Watson’s Jeopardy training?) with virtual assistance skills
 * However, unlike marketing bots, customer service chatbots must be well-grounded. 
 * And the knoweldge base used to "ground" their answers to reality must be kept current.

# Therapy Chatbots
 * Modern therapy chatbots, like Wysa and YourDost, have been built to help displaced tech workers adjust to their new lives
 * They must be entertaining like a conversational chatbot. 
 * They must be informative like a question answering chatbot. And they must be pursuasive like a marketing chatbot. 
 * if they are imbued with self-interest to augment their altruism, these chatbots may be "goal-seeking" and use their marketing and influence skill to get you to come back for additional sessions.

# Hybrid Chatbots
 * build an "objective function" that will take into account the goals of your chatbot when it is choosing between the four approaches, or merely chosing among all the possible responses generated by each of these four approaches.

---

# 1. Grammar (Pattern Matching)
 * used grammars (regular expressions) or pattern matching to trigger responses. 
 * In addition to detecting statements that your bot can respond to, patterns can also be used to extract information from in the incoming text.
 * The information extracted from your users statements can be used to populate a database of knowledge about the user, or about the world in general.
 * extract the name of the person being greeted by the human user. 
 * This helps give the bot "context" for the conversation. 
 * This context can be used to populate a response. 
 * ELIZA
   * programed with a limited set of important words in users statements. 
   * The single most important word in a user’s statement would then trigger selection of a resonse template.
   * These response templates were carefuly designed to emulate the empathy and open-mindedness of a therapist, using "reflexive" psychology. 
   * The key word was often reused in the response. 
   * By replying in a user’s own language, the bot helped build rapport and helped users believe that it was listening.
   *  listening well can be a powerful tool.
 * ALICE
   * developed based on a more general framework for defining these patterns and the resonse templates. Artificial Intelligence Markup Language (AIML)
   * This has since become the defacto standard for defining chatbot and virtual assistant configuration APIs for services such as Pandorabots.

# Artificial Intelligence Markup Language (AIML)
 * AIML is an open standard and there are open source python packages for parsing and "executing" AIML for you chatbot.
 * AIML is a declarative language built on the XML standard.
 * List of {category: {paterns, template}} structures.
 * One limitation of AIML is the kinds of patterns we can match and respond to. An AIML kernel (pattern matcher) only responds when input text matches a pattern "hard coded" by a developer.
 * Restricted RegEx (single * wild card) must match exactly.
 * No fuzzy matches, emoticons, internal punctuation characters, typos, or misspellings can be matched automatically. In AIML you have to manually define "synonyms".
 * In AIML 2.0 you can specify alternative random response templates with square-bracketed lists. 
 * And AIML even has tags for defining variables names, topics, and conditionals.

 ---

* The sophistication of a chatbot built this way grows linearly with the human effort put into it. In fact, as the complexity of these chatbot grows we begin to see diminishing returns on our effort, as the interactions between all the "moving parts" grows and the chatbot behavior becomes harder and harder to predict and debug.

* Data-driven programming is the modern approach to most complex programming challenges these days. How can we use data to program our chatbot?

 # 2. Grounding (context) (listening)
 * create structured knowledge from natural language text (unstructured data) using information extraction.
 * build up a network of relationships or facts just based on reading text, like Wikipedia, or even your own personal journal.
 * This network of logical relationships between things is a knowledge graph or knowledge base that can drive our chatbot’s responses.
 * This knowledge graph can be processed with logical inference compose responses to questions about the world of knowledge contained in the knowledge base. 
 * This knowledge-based approach isn’t limited to answering questions just about the world (or your life). 
 * Your knowledge base can also be populated in real time with facts about an ongoing dialog.
 * Each statement by dialog participants can be used to populate a "theory of mind" knowledge base about what each speaker believes about the world.
 * make several inferences and assumptions to understand and respond to a single statement.
 * we need is a way to "query" the knowledge base to extract the facts we need to populate a response to a user’s statement.
 * check for key question words like "who", "what", "when", "where", "why", and "is" at the beginning of a sentence to classify the type of question. 
 * This would help our chatbot determine the kind of knowledge (node or named entity type) to retrieve from our knowledge graph.

# open knowledge bases
* Wikidata (includes Freebase)
* Open Mind Common Sense (ConceptNet) 
* Cyc
* YAGO
* DBpedia

---

# 3. Retrieval (Search)
 * Another more data-driven approach to "listening" to your user is to search for previous statements in your logs of previous conversations.
 * But, as usual, garbage in means garbage out. Clean data is valuable.
 * To facilitate this search, the dialog corpus should be organized in statement-response pairs.
 * if there is no answer in the knowledgebase then another technique needs to be used.

# Context Challenge
 * The simplest approach is to reuse the response verbatum, without any adjustment.
 * For example, what if someone asked your chatbot "what time is it?" Your chatbot shouldn’t reuse the reply of the human who replied to the best-matched statement in your database.
 * create persona for bot.
 * tag data in the knowlegebase as applicable for types of persona, and only use those based on the bot persona.

* github.com/totalgood/prosocial-chatbot/
* ubuntu dialog corpus

---

# Generative Models

* Autoencoders: Markhov Chains trained to recreate their input sequences
* Restricted Boltzmann Machines (RBMs): Markov Chains trained to minimize an "Energy" functio
* Generative Adversarial Networks (GANs): Statistical models trained to fool a "judge" of good conversation
* Attention Networks

--- 

# Combinaing All 4 approaches
 * we need a modern chatbot framework that is easy to extend and modify and can efficiently run each of these algorithm types in parallel.
 *  We’re going to add a response generator for each of the four approaches using the python examples above. 
 * And then we’re going to add the logic to decide what to actually say by choosing one of the four (or many) responses.

# Will
 * Will is a modern programmer-friendly chatbot framework that can participate in your HipChat and Slack channels as well as others
 * Will uses regular expressions to make matches. 
 * Python itself can be used for any logical conditions you need to evaluate. 
 * And the jinja2 library is used for templating
 * Will still suffers from the same limitations that hold back all pattern-based chatbots (including AIML)--it can’t learn from data, it must be "taught" by the developer writing code for each and every branch in the logic tree.
 * To create a useful app, product managers and developers compose user stories. 
 * A user story describes a sequence of actions performed by a user in interacting with your app and how your app should respond.
 * User stories for a chatbot can often be composed as statements (text messages) that a user might communicate to the bot.
 














---

python - Will
python - ChatterBot
python - PyAiml, aiml, and aiml_bot












---

NB: chatbots that can take actions?