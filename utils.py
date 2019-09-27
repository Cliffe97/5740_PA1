import re
import nltk

START_SYMBOL = '<s>'

def read_preprocess(file):
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



if __name__ == '__main__':
    sen = ' a bbbb a the ffff 355555 apple thebook'
    sen = re.sub(r'\d',' ',sen)
    # sen = re.sub('  ', ' ', sen)
    print(sen)
    sen = sen.split()
    # sen = re.sub(' the ', ' ', sen)
    # sen = re.sub(r'\number', '', sen)
    # sen = re.sub(' the ', '', sen)
    print(sen)