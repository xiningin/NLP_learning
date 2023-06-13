import nltk
from nltk.corpus import reuters
from tqdm import tqdm
from collections import deque

VOCAB_PATH = '../vocab.txt'
TEST_PATH = '../testdata.txt'
PUNCTUATIONS = ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']

def preprocessing(ngram , cate):
    """
    ngram: the number N of N-Gram
    cate: used to choose corpus from reuters

    preprocess all data(vocab , testdata , corpus) for later use
    """
    vocabfile = open(VOCAB_PATH , 'r')
    vocab_list = []
    for line in vocabfile:
        # print(line[:-1]) 
        vocab_list.append(line[:-1]) # delete '\n' after each word

    # read testdata , preprocess it and store it to a list
    testfile = open(TEST_PATH , 'r')
    test_data = []
    for line in testfile:
        # eg:  8	1	We are aware of the seriousnyss of the U.S.
        item = line.split('\t')
        item[2] = nltk.word_tokenize(item[2])
        item[2] = ['<s>'] + item[2] + ['</s>']
        # remove punctuations
        for words in item[2]:
            if words in PUNCTUATIONS:
                item[2].remove(words)
        
        test_data.append(item)

    # preprocess the corpus and generate the count-file of n-gram
    corpus_raw_text = reuters.sents(categories=cate)
    corpus_text = []
    gram_count = {}
    vocab_corpus = []
    for sents in tqdm(corpus_raw_text , desc='Processing the corpus -----'):
        sents = ['<s>'] + sents + ['</s>']
        # remove punctuation
        for words in sents:
            if words in PUNCTUATIONS:
                sents.remove(words)
        corpus_text.extend(sents)

        # count the N-Gram
        for n in range(1 , ngram+1):
            if len(sents) <= n: 
                # not use those short sentences
                continue
            else:
                for i in range(n , len(sents)+1):
                    gram = sents[i-n : i]
                    key = ' '.join(gram)
                    if key in gram_count:
                        gram_count[key] += 1
                    else:
                        gram_count[key] = 1

        vocab_corpus.extend(sents)

    # vocabulary removal repetition  
    vocab_corpus = list(set(vocab_corpus))
    V_len = len(vocab_corpus)

    return vocab_list , test_data , gram_count , vocab_corpus , corpus_text , V_len

END = '$'

def make_trie(vocab):
    """
    vocab: a list of all Uni-Gram 、 Bi-Gram(if have , or it can even have Tri-Gram and so on)

    use dictionary trees to store vocabularies for faster queries
    """
    trie = {}
    for word in vocab:
        t = trie
        for letter in word:
            if letter not in t:
                # create a new letter node
                t[letter] = {}
            # move to the next level(letter)
            t = t[letter]
        # add end node
        t[END] = {}

    return trie

def get_candidate(trie , word , path='' , edit_distance=1):
    """
    trie: prefix tree created by vocabulary
    word: origin word to find the candidate
    edit_distance: times of letter changes

    it will return the candidate list of the error word according to the given edit_distance
    """

    # 
    que = deque([(trie , word , '' , edit_distance)])
    while que:
        trie , word , path , edit_distance = que.popleft()
        if word == '':
        # the first run of the while loop always come here
            if END in trie:
                yield path
            # add(edit operation) a letter to the end of a word
            if edit_distance > 0:
                for k in trie:
                    if k != END:
                        que.appendleft((trie[k] , '' , path+k , edit_distance-1))
        else:
            if word[0] in trie:
                # the first letter is a match
                que.appendleft((trie[word[0]] , word[1:] , path+word[0] , edit_distance))
            # no matter whether the initial letter is matched, the process is as follows
            # generate the candidate by edit in one step
            if edit_distance > 0:
                edit_distance -= 1
                for k in trie.keys() - {word[0] , END}:
                    # use letter k to exchange(edit operation) the origin letter 
                    que.append((trie[k] , word[1:] , path+k , edit_distance))
                    # add(edit operation) letter k in this position
                    que.append((trie[k] , word , path+k , edit_distance))
                # delete(edit operation) this letter
                que.append((trie , word[1:] , path , edit_distance))
                # swap(edit operation) the first two letters of the target word, keeping the end point trie
                if len(word) > 1:
                    que.append((trie , word[1]+word[0]+word[2:] , path , edit_distance))


if __name__ == '__main__':
    # ============================================
    cate = reuters.categories()
    vocab, testdata, gram_count, vocab_corpus, corpus_text, V_len = preprocessing(2, cate)
    print(V_len , len(gram_count))

    # ============================================
    trie = make_trie(vocab)
    # candidates = get_candidate(trie , 'miney' , path='' , edit_distance=1) # money mined mines miner mine
    candidates = get_candidate(trie , 'wsohe' , path='' , edit_distance=2) # she shoe sole sore some soe swore wrote whole whore whose whoe whoe wove wore woke woe wohd
    for candi in candidates:
        print(candi , end=' ')
    print()

