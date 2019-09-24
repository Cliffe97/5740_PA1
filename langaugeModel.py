import math
import operator


class Bigram:
    def __init__(self):
        self.dict1 = dict()
        self.dict2 = dict()
        self.count = 0

    def train(self, file):
        with open(file,'r') as f:
            for line in f:
                prev = '.'
                line = line.split(' ')
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
        self.dict1['/unk'] = 0;
        with open(file, 'r') as f:
            for line in f:
                prev = '.'
                line = line.split(' ')
                for word in line:
                    if word in self.dict1:
                        self.dict1[word] += 1
                    else:
                        self.dict1[word] = 0
                        self.dict1['/unk'] +=1;

                    if prev + word in self.dict2:
                        self.dict2[prev + word] += 1
                    else:
                        self.dict2[prev + word] = 1
                    prev = word

    def train_with_topM(self, file, M):
        with open(file, 'r') as f:
            for line in f:
                line = line.split(' ')
                for word in line:
                    self.count += 1
                    word = word.lower()
                    if word in self.dict1:
                        self.dict1[word] += 1
                    else:
                        self.dict1[word] = 1

        items = sorted(dict.items(), key=operator.itemgetter(1))
        disposed = items[M:]
        self.dict1['/unk'] = 0
        for key, value in disposed:
            self.dict1['/unk'] += value
            del self.dict1[key]

        with open(file, 'r') as f:
            for line in f:
                prev = '.'
                line = line.split(' ')
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
                line = line.split(' ')
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
    sentence = "we love the location and proximity to everything . The staff was very friendly and courteous . They were so nice to our 2.5 year old boy . got his backpack full of goodies the moment we arrived . We got free wifi and morning drinks by signing up for select guest program . Ca n't beat that ! the only minor issue is the elevator . we have to take 2 separate elevator trips to get to our room . It got a little annoying when we were going in and out often . Otherwise , it was a great stay !"

    file_T = 'train/truthful.txt'

    file_D = 'train/deceptive.txt'
    sentence = sentence.lower().split(" ")
    model_T = Bigram()
    model_D = Bigram()
    model_D.train_with_first_OOV(file_D)
    model_T.train_with_first_OOV(file_T)
    print(model_T.test_with_addK(1, sentence))
    print(model_D.test_with_addK(1, sentence))
    # print(model_T.test(sent2))
    # model_D = Unigram(file_D)