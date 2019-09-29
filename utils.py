import re
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

START_SYMBOL = '<s>'

def lm_preprocess(file):
    stemmer = EnglishStemmer()
    corpus = []
    with open(file,'r') as f:
        for review in f:
            review = re.sub(' \? ', ' ? '+START_SYMBOL+' ', review)
            review = re.sub(' ! ', ' ! ' + START_SYMBOL + ' ', review)
            review = re.sub(' \. ', ' . ' + START_SYMBOL + ' ', review)

            review_tokenized = word_tokenize(review)
            review_tagged = pos_tag(review_tokenized)
            review = START_SYMBOL + ' ' + review
            review = review.split()
            offset = 1
            for i in range(1,len(review)):
                if review[i] == review_tagged[i-offset][0]:
                    # print(review[i], " ", review_tagged[i - offset])
                    if (review_tagged[i-offset][1] == "CC"):
                        review[i] = "<CC>"
                    elif (review_tagged[i-offset][1] == "CD"):
                        review[i] = "<CD>"
                    elif (review_tagged[i-offset][1] == "IN"):
                        review[i] = "<IN>"
                    elif (review_tagged[i-offset][1] == "NNP"):
                        review[i] = "<NNP>"
                    elif (review_tagged[i-offset][1] == "PRP"):
                        review[i] = "<PRP>"
                    elif (review_tagged[i-offset][1] == "PRP$"):
                        review[i] = "<PRP$>"
                    elif (review_tagged[i-offset][1] == "WRB"):
                        review[i] = "<WRB>"
                    review[i] = stemmer.stem(review[i])
                    review[i] = review[i].lower()
                else:
                    # print(review[i], " ", review_tagged[i - offset],"*********")
                    offset += 1
            print(review)
            corpus.append(review)
    return corpus

def perplexity(log_probs, lengths):
    result =  (-log_probs)/lengths
    return 2**result





if __name__ == '__main__':

    review = 'We have been for the first time in Chicago and stayed in the Swissotel for five nights , a wonderful place.Room was on the 29th floor , absolutely clean and nice and a breathtaking lakeview.Breakfast perfect and service really good.People were always helpful.It \'s a little bit far from Magnificent mile and the loop , but in front of the hotel there are buses ( number6 ) waiting there next turn , not a real bus stop , but drivers are frindly and you can get on the bus.Next time going to Chicago we will again stay in the Swissotel . '

    pass