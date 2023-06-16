import numpy as np

"""
The overall idea of this model is to use the Bayesian probability combined with the Markov chain 
to derive the posterior probability based on the prior probability 

prior probability: P(w|c)
posterior probability: P(c|w)
c is the correct word 
w is the word in question
"""


# P(c)
def language_model(gram_count , V_len , candidate_data , ngram , lamd):
    """
        gram_count: the dict of all vocabulary(or vocab composition) frequency
        V_len: length of the vocabulary(or vocab composition)
        candidate_data: the candidate data used to calculate probability
        ngram: the number N of N-Gram
        lamd: a temperature coefficient for backoff smoothing 

        given a part of sentence , predict the probability
    """
    if ngram == 1: # Uni-Gram
        # for each token , increment by 1 for Laplace smoothing
        key = candidate_data[0]
        if key in gram_count:
            pi = (gram_count[key]+1) / V_len
        else:
            pi = 1 / V_len
        return np.log(pi)
    else:
        # backoff smoothing
        keys = ' '.join(candidate_data)
        keys_pre = ' '.join(candidate_data[:-1])

        if keys in gram_count and keys_pre in gram_count:
            pi = (gram_count[keys] + lamd) / (gram_count[keys_pre] + lamd*V_len)
        elif keys in gram_count:
            pi = (gram_count[keys] + lamb) / (lamd*V_len)
        elif keys_pre in gram_count:
            pi = lamd / (gram_count[keys_pre] + lamd*V_len)
        else:
            pi = 1 / V_len
        
        return np.log(pi)

    
# P(w|c)
def channel_model(x , y , wrong_reason , corpus , add_matrix , sub_matrix , rev_matrix , del_matrix):
    """
    x: wrong letter in front
    y: wrong letter in back(xy combination was expected to be counted in XXXconfusion.data)
    wrong_reason: the wrong reason of the wrong vocab
    corpus: ....
    add_matrix: read from addconfusion.data , holding the frequencies of wrong letter combination
    sub_matrix: read from subconfusion.data , holding the frequencies of wrong letter combination
    rev_matrix: read from revconfusion.data , holding the frequencies of wrong letter combination
    del_matrix: read from delconfusion.data , holding the frequencies of wrong letter combination

    method to calculate channel model probability for errors
    """
    print(corpus)
    corpus_str = ' '.join(corpus)
    if wrong_reason == 'add':
        if x+y in add_matrix and corpus_str.count(' '+y) and corpus_str.count(x):
            if x == '#':
                return (add_matrix[x+y] + 1) / corpus_str.count(' ' + y)
            else:
                return (add_matrix[x+y] + 1) / corpus_str.count(x)
        else:
            return 1 / len(corpus)
    
    if wrong_reason == 'sub':
        if x+y in sub_matrix and corpus_str.count(y):
            return (sub_matrix[x+y] + 1) / corpus_str.count(y)
        elif x+y in sub_matrix:
            return (sub_matrix[x+y] + 1) / len(corpus)
        elif corpus_str.count(y):
            return 1 / corpus_str.count(y)
        else:
            return 1 / len(corpus)

    if wrong_reason == 'rev':
        if x+y in rev_matrix and corpus_str.count(x+y):
            return rev_matrix[x+y] + 1 / corpus_str.count(x+y)
        elif x+y in rev_matrix:
            return rev_matrix[x+y] / len(corpus)
        elif corpus_str.count(x+y):
            return 1 / corpus_str.count(x+y)
        else:
            return 1 / len(corpus)

    if wrong_reason == 'del':
        if x+y in del_matrix and corpus_str.count(x+y):
            return (del_matrix[x+y] + 1) / corpus_str.count(x+y)
        elif x+y in del_matrix:
            return (del_matrix[x+y] + 1) / len(corpus)
        elif corpus_str.count(x+y):
            return 1 / corpus_str.count(x+y)
        else:
            return 1 / len(corpus)              
            
