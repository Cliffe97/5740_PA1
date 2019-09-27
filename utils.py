import re
from nltk.stem.snowball import EnglishStemmer

START_SYMBOL = '<s>'

def preprocess(file):
    stemmer = EnglishStemmer()
    corpus = []
    with open(file,'r') as f:
        for review in f:
            review = START_SYMBOL +' '+ review
            review = re.sub(' a ', ' ', review)
            review = re.sub(' an ', ' ', review)
            review = re.sub(' the ', ' ', review)
            review = re.sub(' is ', ' ', review)
            review = re.sub(' are ', ' ', review)
            review = re.sub(' was ', ' ', review)
            review = re.sub(' were ', ' ', review)
            review = re.sub('-', ' ', review)
            review = re.sub(' ? ', ' ? '+START_SYMBOL+' ', review)
            review = re.sub(' ! ', ' ! ' + START_SYMBOL + ' ', review)
            review = re.sub(' . ', ' . ' + START_SYMBOL + ' ', review)
            review = review.split()
            for i in range(len(review)):
                review[i] = stemmer.stem(review[i])
            corpus.append(review)
    return corpus

def perplexity(log_probs, lengths):
    result =  (-log_probs)/lengths
    return 2**result

if __name__ == '__main__':
    import numpy as np
    a = np.array([2,2,2,2]).reshape((4,1))
    b = np.array([1, 1, 1, 1]).reshape((4,1))
    print(perplexity(a,b))