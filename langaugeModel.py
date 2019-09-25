import math
import operator
import numpy as np


class Bigram:
    def __init__(self):
        self.dict1 = dict()
        self.dict2 = dict()
        self.count = 0

    def train(self, file):
        with open(file,'r') as f:
            for line in f:
                prev = '.'
                line = line.lower().split(' ')
                for word in line:
                    if word in self.dict1:
                        self.dict1[word] +=1
                    else:
                        self.dict1[word] = 1

                    if prev+word in self.dict2:
                        self.dict2[prev + word] += 1
                    else:
                        self.dict2[prev + word] =1
                    prev = word
        # print(self.dict1)
        # print(self.dict2)

    def train_with_first_OOV(self, file):
        self.dict1['/unk'] = 0
        with open(file, 'r') as f:
            for line in f:
                prev = '.'
                line = line.lower().split(' ')
                for word in line:
                    if word in self.dict1:
                        self.dict1[word] += 1
                    else:
                        self.dict1[word] = 0
                        self.dict1['/unk'] +=1

                    if prev + word in self.dict2:
                        self.dict2[prev + word] += 1
                    else:
                        self.dict2[prev + word] = 1
                    prev = word


    def train_with_topM(self, file, M):
        with open(file, 'r') as f:
            for line in f:
                line = line.lower().split(' ')
                for word in line:
                    self.count += 1
                    word = word.lower()
                    if word in self.dict1:
                        self.dict1[word] += 1
                    else:
                        self.dict1[word] = 1

        items = sorted(self.dict1.items(), key=operator.itemgetter(1))
        disposed = items[M:]
        self.dict1['/unk'] = 0
        for key, value in disposed:
            self.dict1['/unk'] += value
            del self.dict1[key]

        with open(file, 'r') as f:
            for line in f:
                prev = '.'
                line = line.lower().split(' ')
                for word in line:
                    if word not in self.dict1:
                        word = '/unk'
                    if prev not in self.dict1:
                        prev = '/unk'

                    if prev + word in self.dict2:
                        self.dict2[prev + word] += 1
                    else:
                        self.dict2[prev + word] = 1
                    prev = word
        print(len(self.dict1))

    def test_corpus(self, file, k):
        res = []
        with open(file, 'r') as f:
            for line in f:
                line = line.lower().split(' ')
                res.append(self.test_with_addK(k, line))
        return res


    def test(self,review):
        prob = 0
        prev = '.'
        for word in review:
            if prev+word in self.dict2:
                print(prev+word)
                prob += math.log(self.dict2[prev+word] / self.dict1[word], 2)
            prev = word
        return 2 ** prob

    def test_with_addK(self, k, review):
        prob = 0
        prev = '.'
        for word in review:
            if word not in self.dict1:
                word = '/unk'
            if prev not in self.dict1:
                prev = '/unk'

            print(prev + word)
            if prev + word in self.dict2:
                # print(prev + word)
                prob += math.log((self.dict2[prev + word] + k) / (self.dict1[word] + k * len(self.dict1)), 2)
            else:
                prob += math.log(k / (self.dict1[word] + k * len(self.dict1)), 2)
            prev = word

        return 2 ** prob

class Unigram:
    def __init__(self,file):
        self.dictionary = dict()
        self.count = 0
        self.train(file)
        print(self.dictionary)



    def train(self,file):
        with open(file,'r') as f:
            for line in f:
                line = line.lower().split(' ')
                for word in line:
                    self.count+=1
                    word = word.lower()
                    if word in self.dictionary:
                        self.dictionary[word]+=1
                    else:
                        self.dictionary[word] = 1

    def test(self, review):
        prob = 0
        for word in review:
            prob += math.log(self.dictionary[word] / self.count, 2)
        return 2**prob



if __name__ == '__main__':
    sentence = "I want to issue a travel-warning to folks who might sign up for the weekend deal they offer through travelzoo from time to time : The deal says `` free breakfast '' included in the price . However , what they do n't tell you , is that the breakfast consists of a cup of coffee and a bisquit ( or two ) ! Moreover , you need to ask for these `` tickets '' at the lobby when you check in - they wo n't give them to you automatically ! We stayed there over Christmas '03 , and we , and I noticed several guests who bought the same package , had a rather unpleasant experience ! The hotel is nice though , if you do n't consider their lousy service !"
    file_T = 'train/truthful.txt'
    file_D = 'train/deceptive.txt'
    sentence = sentence.lower().split(" ")
    model_T = Bigram()
    model_D = Bigram()
    model_D.train_with_first_OOV(file_D)
    model_T.train_with_first_OOV(file_T)
    print(model_T.test_with_addK(1,sentence))
    print(model_D.test_with_addK(1,sentence))

    # res1 = np.array(model_T.test_with_addK('validation/truthful.txt', 1))
    # res2 = np.array(model_D.test_corpus('validation/truthful.txt', 1))
    # a = res1>=res2
    # unique_elements, counts_elements = np.unique(a, return_counts=True)
    # print(res1)
    # print(res2)
    #
    # print(unique_elements)
    # print(counts_elements)

