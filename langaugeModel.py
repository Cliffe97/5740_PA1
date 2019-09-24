import math

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
    sentence = "I stayed for four nights while attending a conference . The hotel is in a great spot - easy walk to Michigan Ave shopping or Rush St. ."
    file_T = 'train/truthful.txt'

    file_D = 'train/deceptive.txt'
    sentence = sentence.lower().split(" ")
    model_T = Bigram()
    model_D = Bigram()
    model_D.train(file_D)
    model_T.train(file_T)
    print(model_T.test(sentence))
    print(model_D.test(sentence))
    # print(model_T.test(sent2))
    # model_D = Unigram(file_D)