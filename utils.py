def preprocess(file):
    res = []
    with open(file, 'r') as f:
        for line in f:
            line = line.lower().split(' ')
            res.append(line)
    return res


# res = preprocess('validation/truthful.txt')
# for ele in res:
#     print(ele)
