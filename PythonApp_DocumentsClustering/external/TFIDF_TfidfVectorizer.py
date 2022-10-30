from pickletools import float8
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy


corpus = [
    "The car is driven on the road",
    "The truck is driven on the highway"
    ]

#Initialize Object
"""
Pws upologizetai to TFIDF
tf(t,d) = poses fores emfanizetai to word-t mesa sto document d, 
        den diairoume me to sunolo twn word tou d

idf(t,D) = ln ( (1 + D) / (1 + df))
        D = o arithmos twn documents , count(docouments)
        df = poses fores emfanizetai to t sto D dhladh
            An uparxei(den mas noiazei poses fores uparxei) to t sto document[i] tote df++

"""
vectorizer = TfidfVectorizer(
    lowercase = True, #all corpus text to lower case
    
    #max_df = 1.0, #if a word exist in more than 80% of documents then we remove it
    
    #min_df = 1, #if a word exist less than 1 time into corpus, then we remove it
    
    use_idf = True, #Use or not idf function 
    
    #smooth_idf=True, #True:idf = log(n/(1+df))  False: idf = log(n/df) + 1
    
    #sublinear_tf = False, #True: replace tf with 1 + log(tf)
    
    #max_features = 10, #how many features it will keep 
    
    #stop_words = ['is','the','on'] #Remove string from this list or stop_words = "english" then complete automatically the list with einglish stop words
    
    #ngram_range=(1,3) # features na theorountai apo 1 ews 3 words(kathe dunati triada) ['car','car is', 'car is driven', 'driven' , 'driven on', 'driven on the' ....
    
    
    #norm=None 
    #norm = 'l2' kanoume kanonikopoihsh tou telikou dianusmatos tfidf pou exoume 
    #   dhladh pernoume to mh kanonikopoihmeno dianusma  kai diairoume kathe timh 
    #   tou me th posothta Sqrt(Sum(tfidt[i]^2)) = metro tou dianusmatos
    #   meta tha isxuei Sum(tfidt[i]^2) = 1
    #   sto sklearn anaferetai oti: The cosine similarity between two vectors is their dot product when l2 norm has been applied.
    norm=None

    )



# Vectors
docVectors = vectorizer.fit_transform(corpus)

#type:numpy.ndarray | distinct words among documents ['car', 'driven', 'is', ...] 
featureNames = vectorizer.get_feature_names_out()

#type:tuple | vectrors matrix dimensions ex. (2,8) 2 documents, 8 distict words found in documents 
vectorMatrixShape = docVectors.shape
#type: numpy.matrix can convert to list(2D) | document vectors [[0.4, 0.3, 0.0, ...],[0.0, 0.3, 0.4, ...]] 2x8
dense = docVectors.todense()
denseList = dense.tolist()


# Tokenize a specific Document  ['the', 'truck', 'is', 'driven', 'on', 'the', 'highway']
analyze = vectorizer.build_analyzer()
tokenizedDocument = analyze("The truck is driven on the highway")
 

tf_idf = pd.DataFrame(docVectors.toarray() ,columns=vectorizer.get_feature_names_out())
print(tf_idf)



