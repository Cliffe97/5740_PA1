# import sklearn
import csv
import math
import operator
import numpy as np
from utils import lm_preprocess



class NaiveBayes:
    def __init__(self):

        self.tdict = dict()
        self.ddict = dict()

    def train(self, reviews_T, reviews_D):
        for review in reviews_T:
            for word in review:
                if word in self.tdict:
                    self.tdict[word] += 1
                else:
                    self.tdict[word] = 1
        for review in reviews_D:
            for word in review:
                if word in self.ddict:
                    self.ddict[word] += 1
                else:
                    self.ddict[word] = 1
        print(self.ddict)

if __name__ == '__main__':
    model = NaiveBayes()
    train_file_T = 'train/truthful.txt'
    train_file_D = 'train/deceptive.txt'
    reviews_T = preprocess(train_file_T)
    reviews_D = preprocess(train_file_D)
    model.train(reviews_T,reviews_D)






