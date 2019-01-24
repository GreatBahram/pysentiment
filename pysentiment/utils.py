"""
This module contains methods to tokenize sentences.
"""
import abc
import re

import nltk

class BaseTokenizer(metaclass=abc.ABCMeta):
    """
    An abstract class for tokenize text.
    """

    @abc.abstractmethod
    def tokenize(self, text):
        """Return tokenized temrs.
        
        :type text: str
        
        :returns: list 
        """
        pass


class Tokenizer(BaseTokenizer):
    """
    The default tokenizer for ``pysentiment``, which only takes care of words made up of ``[a-z]+``.
    The output of the tokenizer is stemmed by ``nltk.PorterStemmer``. 
    
    The stoplist from https://www3.nd.edu/~mcdonald/Word_Lists.html is included in this
    tokenizer. Any word in the stoplist will be excluded from the output.
    """
    
    def __init__(self):
        self._stemmer = nltk.PorterStemmer()
        self._stopwords = self.get_stopwords()
        
    def tokenize(self, text, stem=False):
        tokens = []
        for token in nltk.tokenize.word_tokenize(text.lower()):
            if stem:
                token = self._stemmer.stem(token)
            if not token in self._stopwords:
                tokens.append(token)
        return tokens
        
    def get_stopwords(self):
        return nltk.corpus.stopwords.words('english')
