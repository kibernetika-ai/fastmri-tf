import numpy as np
def norm_text(v,use_pace=' '):
    s = []
    space = False
    for i in v.lower():
        if i.isalnum():
            s.append(i)
            space = (i==' ')
        elif not space:
            s.append(use_pace)
            space = True
    return ''.join(s)


def dictionary_size(word_index):
    return len(word_index)-2

def labels(word_index, text):
    text = norm_text(text)
    print(text)
    l = np.zeros(dictionary_size(word_index), dtype=np.float32)
    for k,i in word_index.items():
        if k=='<start>':
            continue
        if k=='<end>':
            continue
        k = k.replace('_',' ')
        if k in text:
            l[i-1]=1
    return l

def tokenize(word_index, text):
    text = norm_text(text)
    for k,i in word_index.items():
        text = text.replace(k.replace('_',' '),k)

    tokens = []
    for t in text.split(' '):
        v = word_index.get(t, None)
        if v is not None:
            tokens.append(v)

    return tokens

def dictionary(file_name):
    word_index = {}
    max_index = 0
    with open(file_name, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            p = line.split(',')
            if len(p) != 2:
                continue
            index = int(p[0])
            max_index = max(index, max_index)
            word_index[norm_text(p[1],'_')] = index
    word_index['<end>'] = max_index + 1
    word_index['<start>'] = 0
    return word_index