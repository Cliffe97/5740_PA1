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



if __name__ == '__main__':
    sen = ' a bbbb a the ffff 355555 apple thebook'
    sen = re.sub(r'\d',' ',sen)
    # sen = re.sub('  ', ' ', sen)
    print(sen)
    stemmer = EnglishStemmer()
    a = ['aaa', 'bbb']
    for i in range(len(a)):
        a[i] = 'vvv'

    print(a)