from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

corpus = [
    "The car is driven on the road",
    "The truck is driven on the highway"
    ]


cv = CountVectorizer()
word_count_vector = cv.fit_transform(corpus)
tf = pd.DataFrame(word_count_vector.toarray(), columns=cv.get_feature_names_out())
print(tf)

tfidf_transformer = TfidfTransformer(
    use_idf = True, #Use or not idf function 
    
    #smooth_idf=True, #True:idf = log(n/(1+df))  False: idf = log(n/df) + 1
    
    sublinear_tf = False, #True: replace tf with 1 + log(tf)

    #norm=None 
    #norm = 'l2' kanoume kanonikopoihsh tou telikou dianusmatos tfidf pou exoume 
    #   dhladh pernoume to mh kanonikopoihmeno dianusma  kai diairoume kathe timh 
    #   tou me th posothta Sqrt(Sum(tfidt[i]^2)) = metro tou dianusmatos
    #   meta tha isxuei Sum(tfidt[i]^2) = 1
    #   sto sklearn anaferetai oti: The cosine similarity between two vectors is their dot product when l2 norm has been applied.
    norm=None
    )
X = tfidf_transformer.fit_transform(word_count_vector)
idf = pd.DataFrame({'feature_name':cv.get_feature_names_out(), 'idf_weights':tfidf_transformer.idf_})
print(idf)

tf_idf = pd.DataFrame(X.toarray() ,columns=cv.get_feature_names_out())
print(tf_idf)
