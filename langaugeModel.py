import math
import operator
import numpy as np
from utils import preprocess

class Bigram:
    def __init__(self):
        self.dict1 = dict()
        self.dict2 = dict()
        self.count = 0

    def train(self, reviews):
        for review in reviews:
            prev = '.'
            for word in review:
                if word in self.dict1:
                    self.dict1[word] += 1
                else:
                    self.dict1[word] = 1

                if prev + word in self.dict2:
                    self.dict2[prev + word] += 1
                else:
                    self.dict2[prev + word] = 1
                prev = word

    def train_with_first_OOV(self, reviews):
        self.dict1['/unk'] = 0
        for review in reviews:
            prev = '.'
            for word in review:
                if word in self.dict1:
                    self.dict1[word] += 1
                else:
                    self.dict1[word] = 0
                    self.dict1['/unk'] += 1

                if prev + word in self.dict2:
                    self.dict2[prev + word] += 1
                else:
                    self.dict2[prev + word] = 1
                prev = word

    def train_with_topM(self, reviews, M):
        for review in reviews:
            for word in review:
                self.count += 1
                word = word.lower()
                if word in self.dict1:
                    self.dict1[word] += 1
                else:
                    self.dict1[word] = 1

        # select top M items to remain in word dictionary
        items = sorted(self.dict1.items(), key=operator.itemgetter(1))
        disposed = items[M:]
        self.dict1['/unk'] = 0
        for key, value in disposed:
            self.dict1['/unk'] += value
            del self.dict1[key]

        # train bi-gram with '/unk' tag
        for review in reviews:
            prev = '.'
            for word in review:
                if word not in self.dict1:
                    word = '/unk'
                if prev not in self.dict1:
                    prev = '/unk'

                if prev + word in self.dict2:
                    self.dict2[prev + word] += 1
                else:
                    self.dict2[prev + word] = 1
                prev = word


    def test_corpus(self, reviews, k):
        res = []
        for review in reviews:
            res.append(self.test_with_addK(k, review))
        return res


    def test(self,review):
        prob = 0
        prev = '.'
        for word in review:
            if prev+word in self.dict2:
                # print(prev+word)
                prob += math.log(self.dict2[prev+word] / self.dict1[word], 2)
            prev = word
        return prob
        # return 2**prob

    def test_with_addK(self, k, review):
        prob = 0
        prev = '.'
        for word in review:
            if word not in self.dict1:
                word = '/unk'
            if prev not in self.dict1:
                prev = '/unk'

            # print(prev + word)
            if prev + word in self.dict2:
                # print(prev + word)
                prob += math.log((self.dict2[prev + word] + k) / (self.dict1[word] + k * len(self.dict1)), 2)
                # print(prob)
            else:
                # print("-------")
                # print(prob)
                prob += math.log(k / (self.dict1[word] + k * len(self.dict1)), 2)
            prev = word

        return prob
        # return 2 ** prob

class Unigram:
    def __init__(self,reviews):
        self.dictionary = dict()
        self.count = 0
        self.train(reviews)
        # print(self.dictionary)



    def train(self,reviews):
        for review in reviews:
            for word in review:
                self.count += 1
                word = word.lower()
                if word in self.dictionary:
                    self.dictionary[word] += 1
                else:
                    self.dictionary[word] = 1

    def test(self, review):
        prob = 0
        for word in review:
            prob += math.log(self.dictionary[word] / self.count, 2)
        return prob
        # return 2 ** prob


if __name__ == '__main__':

    # TRAINING
    train_file_T = 'train/truthful.txt'
    train_file_D = 'train/deceptive.txt'
    reviews_T = preprocess(train_file_T)
    reviews_D = preprocess(train_file_D)
    model_T = Bigram()
    model_D = Bigram()

    # train option 1
    model_T.train_with_first_OOV(reviews_T)
    model_D.train_with_first_OOV(reviews_D)

    # # train option 2
    # m = 2000
    # model_T.train_with_topM(reviews_T, m)
    # model_D.train_with_topM(reviews_D, m)

    k=0.01
    # TESTING against truthful.txt
    test_file = 'validation/truthful.txt'
    reviews_test = preprocess(test_file)
    res1 = np.array(model_T.test_corpus(reviews_test, k))
    res2 = np.array(model_D.test_corpus(reviews_test, k))
    ans = res1 > res2
    unique_elements, counts_elements = np.unique(ans, return_counts=True)
    print("testing truthful.txt")
    print(unique_elements)
    print(counts_elements)

    # TESTING against deceptive.txt
    test_file = 'validation/deceptive.txt'
    reviews_test = preprocess(test_file)
    res1 = np.array(model_T.test_corpus(reviews_test,k))
    res2 = np.array(model_D.test_corpus(reviews_test,k))
    ans = res1 > res2
    unique_elements, counts_elements = np.unique(ans, return_counts=True)
    print("\ntesting deceptive.txt")
    print(unique_elements)
    print(counts_elements)