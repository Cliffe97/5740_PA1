import numpy as np
from nltk.stem.snowball import EnglishStemmer
from sklearn.naive_bayes import MultinomialNB

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

if __name__ == '__main__':

    train_file_T = 'train/truthful.txt'
    train_file_D = 'train/deceptive.txt'
    valid_file_T = 'validation/truthful.txt'
    valid_file_D = 'validation/deceptive.txt'
    prepro = NB_Preprocessor()
    train_X, train_Y = prepro.preprocess_train(train_file_T, train_file_D)
    model = MultinomialNB()
    model.fit(train_X, train_Y)
    test_Y_T = prepro.preprocess_test(valid_file_T)
    test_Y_D = prepro.preprocess_test(valid_file_D)
    res_T = model.predict(test_Y_T)
    unique_elements, counts_elements = np.unique(res_T == 0, return_counts=True)
    print("testing truthful.txt")
    print(unique_elements)
    print(counts_elements)

    res_D = model.predict(test_Y_D)
    unique_elements, counts_elements = np.unique(res_D == 1, return_counts=True)
    print("testing truthful.txt")
    print(unique_elements)
    print(counts_elements)






