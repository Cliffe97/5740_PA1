import csv
import math
import operator
import numpy as np
from utils import lm_preprocess
from utils import perplexity

UNKNOWN_SYMBOL = '/unk'
START_SYMBOL = '<s>'

class Bigram:
    def __init__(self):
        self.uni_dict = dict()
        self.bi_dict = dict()

    def train(self, reviews):
        for review in reviews:
            prev = None
            for word in review:
                if word == START_SYMBOL:
                    prev = None
                self.uni_dict[word] = self.uni_dict.get(word, 0) + 1
                if prev is not None:
                    self.bi_dict[(prev, word)] = self.bi_dict.get((prev, word), 0) + 1

                prev = word

    # we treat the first occurrence of each word type as unknown
    def train_with_first_OOV(self, reviews):
        self.uni_dict[UNKNOWN_SYMBOL] = 0

        for review in reviews:
            prev = None
            for word in review:
                if word in self.uni_dict:
                    self.uni_dict[word] += 1
                else:
                    self.uni_dict[word] = 0
                    self.uni_dict[UNKNOWN_SYMBOL] += 1

                if prev is not None:
                    self.bi_dict[(prev, word)] = self.bi_dict.get((prev, word), 0) + 1
                prev = word

    # we only keep top M most frequent word types and treat every other word as unknown
    def train_with_topM(self, reviews, M):
        for review in reviews:
            for word in review:
                self.uni_dict[word] = self.uni_dict.get(word, 0) + 1

        # select top M items to remain in word dictionary
        items = sorted(self.uni_dict.items(), key=operator.itemgetter(1))
        disposed = items[M:]
        self.uni_dict[UNKNOWN_SYMBOL] = 0
        for key, value in disposed:
            self.uni_dict[UNKNOWN_SYMBOL] += value
            del self.uni_dict[key]

        # train bi-gram with '/unk' tag
        for review in reviews:
            prev = None
            for word in review:
                if word not in self.uni_dict:
                    word = UNKNOWN_SYMBOL

                if prev is not None:
                    if prev not in self.uni_dict:
                        prev = UNKNOWN_SYMBOL
                    self.bi_dict[(prev, word)] = self.bi_dict.get((prev, word), 0) + 1

                prev = word


    def test_corpus(self, reviews, k):
        res = []
        for review in reviews:
            res.append(self.test_with_Ksmoothing(k, review))
        return res


    # def test(self,review):
    #     prob = 0
    #     prev = None
    #     for word in review:
    #         if prev = None
    #         if prev+word in self.bi_dict:
    #             # print(prev+word)
    #             prob += math.log(self.bi_dict[prev + word] / self.uni_dict[word], 2)
    #         prev = word
    #     return prob
        # return 2**prob

    def test_with_Ksmoothing(self, k, review):
        prob = 0
        prev = None
        # print(review)
        for word in review:
            if word not in self.uni_dict:
                word = UNKNOWN_SYMBOL

            if prev is not None:
                if prev not in self.uni_dict:
                    prev = UNKNOWN_SYMBOL
                prob += math.log((self.bi_dict.get((prev, word),0)+k) / (self.uni_dict[word] + k * len(self.uni_dict)),
                                     2)
            prev = word

        return prob
        # return 2 ** prob

class Unigram:
    def __init__(self,reviews):
        self.uni_dict = dict()
        self.count = 0
        self.train(reviews)
        self.uni_dict[UNKNOWN_SYMBOL] = 0

    def train(self,reviews):
        for review in reviews:
            for word in review:
                self.count += 1
                word = word.lower()
                if word in self.uni_dict:
                    self.uni_dict[word] += 1
                else:
                    self.uni_dict[word] = 1

    def train_with_first_OOV(self, reviews):
        for review in reviews:
            for word in review:
                if word in self.uni_dict:
                    self.uni_dict[word] += 1
                else:
                    self.uni_dict[word] = 0
                    self.uni_dict[UNKNOWN_SYMBOL] += 1

    def train_with_topM(self, reviews, M):
        for review in reviews:
            for word in review:
                if word in self.uni_dict:
                    self.uni_dict[word] = self.uni_dict.get(word, 0)+1

        items = sorted(self.uni_dict.items(), key=operator.itemgetter(1))
        disposed = items[M:]
        for key, value in disposed:
            self.uni_dict[UNKNOWN_SYMBOL] += disposed[key]
            del self.uni_dict[key]


    def test(self, review):
        prob = 0
        for word in review:
            prob += math.log(self.uni_dict[word] / self.count, 2)
        return prob
        # return 2 ** prob


if __name__ == '__main__':

    # TRAINING
    train_file_T = 'train/truthful.txt'
    train_file_D = 'train/deceptive.txt'
    reviews_T = lm_preprocess(train_file_T)
    reviews_D = lm_preprocess(train_file_D)
    model_T = Bigram()
    model_D = Bigram()

    # train option 1
    model_T.train_with_topM(reviews_T, 200)
    model_D.train_with_topM(reviews_D, 200)

    # # train option 2
    # m = 2000
    # model_T.train_with_topM(reviews_T, m)
    # model_D.train_with_topM(reviews_D, m)

    k = 1
    # TESTING against truthful.txt
    test_file = 'validation/truthful.txt'
    reviews_test = lm_preprocess(test_file)
    res1 = np.array(model_T.test_corpus(reviews_test, k))
    res2 = np.array(model_D.test_corpus(reviews_test, k))
    lengths = np.array([len(reviews_test[i]) for i in range(len(reviews_test))])
    per1 = perplexity(res1, lengths)
    per2 = perplexity(res2, lengths)
    print(per1)
    print(per2)
    ans = per1 < per2
    unique_elements, counts_elements = np.unique(ans, return_counts=True)
    print("testing truthful.txt")
    print(unique_elements)
    print(counts_elements)

    # TESTING against deceptive.txt
    # test_file = 'validation/deceptive.txt'
    # reviews_test = preprocess(test_file)
    # res1 = np.array(model_T.test_corpus(reviews_test,k))
    # res2 = np.array(model_D.test_corpus(reviews_test,k))
    # ans = res1 > res2
    # unique_elements, counts_elements = np.unique(ans, return_counts=True)
    # print("\ntesting deceptive.txt")
    # print(unique_elements)
    # print(counts_elements)
    #
    #
    # # create .cvs file
    # test_file = 'test/test.txt'
    # reviews_test = preprocess(test_file)
    # length = [len(reviews_test[i]) for i in range(len(reviews_test))]
    # res1 = np.array(model_T.test_corpus(reviews_test, k))
    # res2 = np.array(model_D.test_corpus(reviews_test, k))
    #
    # ans = [['Id', 'Prediction']]
    # for i in range(len(res1)):
    #     if res1[i] <= res2[i]:
    #         ans.append([i, 1])
    #     else:
    #         ans.append([i, 0])
    # # print(ans)
    # # print(len(ans))
    # with open('prediction.csv', 'w') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(ans)
    # csvFile.close()