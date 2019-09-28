import re
from nltk.stem.snowball import EnglishStemmer

START_SYMBOL = '<s>'

def lm_preprocess(file):
    stemmer = EnglishStemmer()
    corpus = []
    with open(file,'r') as f:
        for review in f:
            review = review.lower()
            review = START_SYMBOL +' '+ review
            review = re.sub(' a ', ' ', review)
            review = re.sub(' an ', ' ', review)
            review = re.sub(' the ', ' ', review)
            review = re.sub(' is ', ' ', review)
            review = re.sub(' are ', ' ', review)
            review = re.sub(' was ', ' ', review)
            review = re.sub(' were ', ' ', review)
            review = re.sub('-', ' ', review)
            review = re.sub(' \? ', ' ? '+START_SYMBOL+' ', review)
            review = re.sub(' ! ', ' ! ' + START_SYMBOL + ' ', review)
            review = re.sub(' \. ', ' . ' + START_SYMBOL + ' ', review)
            review = review.split()
            for i in range(len(review)):
                review[i] = stemmer.stem(review[i])
            corpus.append(review)
    return corpus

def perplexity(log_probs, lengths):
    result =  (-log_probs)/lengths
    return 2**result


class NB_Preprocessor:
    def __init__(self):
        self.wordtype_dict = dict()
        self.count = 0

    def preprocess_train(self,truth_file, decp_file):
        normalized_text = self.__setup_vocabulary(truth_file,decp_file)
        train_X = []
        train_Y = []
        for i in range(len(normalized_text)):
            for data in normalized_text[i]:
                for review in data:
                    x = [0]* (self.count)
                    for word in review:
                        x[self.wordtype_dict[word]]+=1
                    train_X.append(x)
                    train_Y.append(i)

        return train_X, train_Y



    def __setup_vocabulary(self, truth_file, decp_file):
        stemmer = EnglishStemmer()
        filenames = [truth_file, decp_file]
        normalized_text = []
        for filename in filenames:
            data = []
            with open(filename, 'r') as file:
                for review in file:
                    record = []
                    review = review.lower()
                    review = review.split()
                    for word in review:
                        word = stemmer.stem(word)
                        record.append(word)
                        if word not in self.wordtype_dict:
                            self.wordtype_dict[word] = self.count
                            self.count += 1
                    data.append(record)
            normalized_text.append(data)
        return normalized_text



if __name__ == '__main__':
    prepro = NB_Preprocessor()
    train_X, train_Y =