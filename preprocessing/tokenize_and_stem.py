import re
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
import nltk

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    w = 0
    while w < len(tokens):
        if tokens[w].lower() in stopwords or tokens[w] == "'s":
            del tokens[w]
            w -= 1
        w += 1
    
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")