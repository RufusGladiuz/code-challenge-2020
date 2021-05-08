from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class LemmaTokenizer(object):
    """ Custom Token Lemmatizer using WordNetLemmatizer.
    
        Args:
            lemmatizer -> A lemmatizer object
            stopwords:list -> List of stopwords to remove from articles
        
    
    """
    
    def __init__(self, lemmatizer=WordNetLemmatizer(),
                 stop_words: list =stopwords.words('english')):
        self._lemma = WordNetLemmatizer()
        self._stpwords = stop_words

    def __call__(self, article):

        article = article.lower()
        article = re.findall("[A-Za-z]+", article)

        article = [token for token in article if token not in self._stpwords]
        return [self._lemma.lemmatize(token) for token in article]
