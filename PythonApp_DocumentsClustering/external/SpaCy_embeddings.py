#Download and import spacy
#Guide https://stackoverflow.com/questions/36835341/pip-is-not-recognized
#Error: pip : The term 'pip' is not recognized as the name of a cmdlet, function, script file, or operable program.
#1)Add pip to Environment Variables 
#    This PC -> Properties -> Advanced System Settings -> Environment Variables ->
#    on System variables clik on Path and then Edit -> New ->
#    Copy/Paste the path where pip exist (C:\Users\George Georgariou\source\repos\PythonApp_DocumentsClustering\PythonApp_DocumentsClustering\VirtualEnv_DocClust\Scripts)
#2)pip install -U pip setuptools wheel
#3)pip install -U spacy

#Download trained pipelines (en_core_web_sm, el_core_news_sm)
#Open terminal here: View -> Terminal
#python -m spacy download en_core_web_lg
#python -m spacy download el_core_news_lg

import spacy

#Language object containing all components and data needed to process text.
nlp = spacy.load('en_core_web_lg')
nlpGr = spacy.load('el_core_news_lg')

#Calling the nlp object on a string of text will return a processed Doc object-container
doc = nlp("Apple is looking at buying U.K. startup for $1 billion ?. Hellow Worlds")

#This Doc components dont need a trained pipeline to produce below components
'''
for token in doc:
    print(token.text, token.pos_, token.dep_)
'''


def useful_token(token):
    """
    Keep useful tokens which have 
       - Part Of Speech tag (POS): ['NOUN','PROPN','ADJ']
       - Alpha(token is word): True
       - Stop words(is, the, at, ...): False
    """
    return token.pos_ in ['NOUN','PROPN','ADJ'] and token.is_alpha and not token.is_stop 

usefull_tokens_list = [token for token in doc if useful_token(token)]
usefull_tokens_generator = (token for token in doc if token.pos_ in useful_token(token))
usefull_tokens_set = {token for token in doc if token.pos_ in useful_token(token)}


# Doc container-object containes sents attributes (sentences of a document)
# where each sent in sents has tokens as attributes
'''
print(f'Document: {doc}')
for sent in doc.sents:
    print(f"Sentensce: {sent}")
    for token in sent:
        print(f"        {token}")
'''

#When we create a Doc object-container, during processing, spaCy first tokenizes the text (token.text)
#After tokenization, spaCy can parse and tag a given Doc. This is where the trained pipeline and 
#its statistical models come in, which enable spaCy to make predictions of which tag or label 
#most likely applies in this context.
'''
for token in doc:
    print(token.text,token.lang_, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
'''

# A named entity is a real world object thats assigned a name for example, a person, acountry
# a products or a book title. SpaCy can recognize various types of a named entity in a document
# by asking the model for a prediction. Because models are statical and strongly depend on the
# examples thew were trained on, this doesnt always work perfectly and might need some tuning
# later, depending on your case.
'''
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
spacy.displacy.serve(doc, style="ent") #browser -> http://localhost:5000/
'''


# Token Word vectors - embeddings
'''
print("Tokens attributes for vectorizing")
doc = nlp("Hello")
for token in doc:
    print(token.text, token.has_vector, token.vector, token.vector_norm, token.is_oov)
    print ("--------------Token Vector-------------")
    print(token.vector)
'''

# Token Word similarity
'''
doc = nlp("Hello Hi")
word1 = doc[0]
word2 = doc[1]
print(f"Words-Tokens similarity: {word1.similarity(word2)}")
'''


# Docoments vectors - embeddings
'''
doc = nlp("Hello Hi")
print("Documents attributes for vectorizing")
print(doc.has_vector, doc.vector_norm)
print ("--------------Document Vector-------------")
print(doc.vector)
'''

# Documents similarity
'''
doc1 = nlp("Hello")
doc2 = nlp("Hi")
print(f"Documents similarity: {doc1.similarity(doc2)}")
'''


