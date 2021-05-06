from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

#TODO: Add Comments

class LemmaTokenizer(object):

    def __init__(self, lemmatizer =  WordNetLemmatizer(),
                stop_words = stopwords.words('english')):
        self.wnl = WordNetLemmatizer()
        self.stpwords = stop_words

    def __call__(self, article):

        article = article.lower()
        article = re.findall("[A-Za-z]+", article)
        
        article = [token for token in article if token not in self.stpwords]
        return [self.wnl.lemmatize(token) for token in article]