import numpy as np
import operator
from nltk.stem.snowball import EnglishStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from utils import lm_preprocess
import fileinput
import csv

class NB_Preprocessor:
    def __init__(self):
        self.wordtype_dict = dict()
        self.count = 0
        self.stemmer = EnglishStemmer()

    def preprocess_train(self,truth_file, decp_file):
        normalized_text = self.__setup_vocabulary(truth_file,decp_file)
        train_X = []
        train_Y = []
        for i in range(len(normalized_text)):
            for review in normalized_text[i]:
                x = [0]* (self.count)
                for word in review:
                    x[self.wordtype_dict[word]]+=1
                train_X.append(x)
                train_Y.append(i)

        return train_X, train_Y

    def preprocess_test(self,test_file):
        test_X = []
        with open(test_file,'r') as file:
            for review in file:
                x = [0]*self.count
                review = review.lower()
                review = review.split()
                for word in review:
                    word = self.stemmer.stem(word)
                    if word in self.wordtype_dict:
                        x[self.wordtype_dict[word]] += 1
                test_X.append(x)
        return test_X

    def __setup_vocabulary(self, truth_file, decp_file):
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
                        word = self.stemmer.stem(word)
                        record.append(word)
                        if word not in self.wordtype_dict:
                            self.wordtype_dict[word] = self.count
                            self.count += 1
                    data.append(record)
            normalized_text.append(data)
        # print(self.wordtype_dict)
        return normalized_text

    def preprocess_bigram_train(self, truthfile, decpfile):
        # vectorizer = CountVectorizer()
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
        corpus = []
        with open(truthfile, 'r') as file:
            for review in file:
                corpus.append(review)
        train_Y = [0] * len(corpus)
        with open(decpfile, 'r') as file:
            for review in file:
                corpus.append(review)

        train_Y.extend([1] * (len(corpus)-len(train_Y)))
        self.vectorizer = bigram_vectorizer
        X_2 = self.vectorizer.fit_transform(corpus).toarray()
        return X_2, train_Y

    def process_test_bigram(self, testfile):
        corpus = []
        with open(testfile, 'r') as file:
            for review in file:
                corpus.append(review)
        return self.vectorizer.transform(corpus)

def parameter_tuning(alphas):
    train_file_T = 'train/truthful.txt'
    train_file_D = 'train/deceptive.txt'
    valid_file_T = 'validation/truthful.txt'
    valid_file_D = 'validation/deceptive.txt'

    prepro = NB_Preprocessor()
    train_X, train_Y = prepro.preprocess_bigram_train(train_file_T, train_file_D)

    test_X_T = np.array(prepro.process_test_bigram(valid_file_T))
    test_X_D = np.array(prepro.process_test_bigram(valid_file_D))
    test_X = np.vstack((test_X_T,test_X_D))

    print(test_X_T)
    test_Y = [0] * test_X_T.shape[0]
    test_Y.extend([1]*test_X_D.shape[0])
    test_Y = np.array(test_Y)

    result = {}
    for alpha in alphas:
        model = MultinomialNB(alpha = alpha)
        model.fit(train_X, train_Y)
        test_Yhat = np.array(model.predict(test_X))
        num_correct = np.sum(test_Yhat == test_Y)
        accuracy = num_correct/len(test_Yhat)
        result[alpha] = accuracy

    items = sorted(result.items(), key=operator.itemgetter(1))
    print(items)

    return items[-1]

def generate_test_csv(best_para):
    train_file_T = 'train/truthful.txt'
    train_file_D = 'train/deceptive.txt'
    test_file = 'test/test.txt'

    prepro = NB_Preprocessor()
    train_X, train_Y = prepro.preprocess_bigram_train(train_file_T, train_file_D)
    test_X = np.array(prepro.process_test_bigram(test_file))

    model = MultinomialNB(alpha=best_para)
    model.fit(train_X, train_Y)
    test_Yhat = np.array(model.predict(test_X))

    ans = [['Id', 'Prediction']]
    for i in range(len(test_Yhat)):
        ans.append([i, test_Yhat[i]])

    with open('nb_bigram_prediction.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(ans)
    csvFile.close()


if __name__ == '__main__':

    # filenames = ['train/deceptive.txt', 'validation/deceptive.txt']
    # with open('train_valid/new_deceptive.txt', 'w') as fout:
    #     fin = fileinput.input(filenames)
    #     for line in fin:
    #         fout.write(line)
    #     fin.close()

    best_para, _ = parameter_tuning([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    generate_test_csv(best_para)
    # nb = NB_Preprocessor()
    #
    # xtrain, ytrain = nb.seetup_bigram_vocabulary('train/truthful.txt', 'train/deceptive.txt')
    # model = MultinomialNB()
    # model.fit(xtrain, ytrain)
    # MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    # test_X = nb.process_test_ngram('test/test.txt')
    # test_Yhat = np.array(model.predict(test_X))
    #
    # ans = [['Id', 'Prediction']]
    # for i in range(len(test_Yhat)):
    #     ans.append([i, test_Yhat[i]])
