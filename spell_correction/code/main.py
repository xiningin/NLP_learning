from data_process import preprocessing , make_trie , get_candidate
from model import language_model , channel_model
from util import edit_finder , load_confusion_matrix
import time
from nltk.corpus import reuters
from tqdm import tqdm

TEST_PATH = '../testdata.txt'
RESULT_PATH = '../result.txt'

def spell_correct(vocab , test_data , gram_count , corpus , V_len , trie , ngram , lamd , add_matrix, sub_matrix, rev_matrix, del_matrix):
    test_file = open(TEST_PATH , 'r')
    data = []

    # process testdata for generate modified file
    for line in test_file:
        item = line.split('\t')
        del item[1]
        data.append('\t'.join(item))

    result_file = open(RESULT_PATH , 'w')

    for item in tqdm(test_data , desc='Spelling correction --'):
        for words in item[2][1:-1]: # use [1:-1] to skip <s> and </s>
            if words in vocab:
                continue
            else:
                if list(get_candidate(trie , words , edit_distance=1)):
                    candidate_list = list(get_candidate(trie, words, edit_distance=1))
                else:
                    candidate_list = list(get_candidate(trie, words, edit_distance=2))
                
                candi_proba = []
                for candidate in candidate_list:
                    if ngram == 1:
                        candi_proba.append(
                            language_model(gram_count , V_len , [candidate] , ngram , lamd)
                        )
                    else:
                        edit = edit_finder(candidate , words)
                        if edit == None:
                            candi_proba.append(
                                language_model(gram_count , len(gram_count) , [candidate] , ngram , lamd)
                            )
                            continue
                        if edit[0] == "Insertion":
                            channel_p = np.log(channelModel(edit[3][0], edit[3][1], 'add', corpus , add_matrix, sub_matrix, rev_matrix, del_matrix))
                        if edit[0] == 'Deletion':
                            channel_p = np.log(channelModel(edit[4][0], edit[4][1], 'del', corpus , add_matrix, sub_matrix, rev_matrix, del_matrix))
                        if edit[0] == 'Reversal':
                            channel_p = np.log(channelModel(edit[4][0], edit[4][1], 'rev', corpus , add_matrix, sub_matrix, rev_matrix, del_matrix))
                        if edit[0] == 'Substitution':
                            channel_p = np.log(channelModel(edit[3], edit[4], 'sub', corpus , add_matrix, sub_matrix, rev_matrix, del_matrix))
                        
                        word_index = item[2][1:-1].index(words)
                        pre_phase = item[2][1:-1][word_index - ngram + 1 : word_index] + [candidate]
                        post_phrase = [candidate] + item[2][1:-1][word_index + 1 : word_index + ngram]
                        
                        p = language_model(gram_count , V_len , pre_phase , ngram , lamd) + \
                            language_model(gram_count, V, post_phase, ngram, lamd)
                        p += channel_p
                        candi_proba.append(p)
                
                index = candi_proba.index(max(candi_proba))
                # start modify origin wrong words
                data[int(item[0]) - 1] = data[int(item[0]) - 1].replace(words , candidate_list[index])
        
        result_file.write(data[int(item[0]) - 1])
    
    result_file.close

if __name__ == '__main__':
    start = time.time()        
    cate = reuters.categories()

    print('Doing preprocessing, computing things. Please wait...')
    vocab, testdata, gram_count, vocab_corpus, corpus_text, V = preprocessing(1, cate)
    add_matrix, sub_matrix, rev_matrix, del_matrix = load_confusion_matrix()
    trie = make_trie(vocab)

    print('Doing Spell Correcting...')
    lamd = 0.01  # add-lambda smoothing
    spell_correct(vocab, testdata, gram_count, corpus_text, V, trie, 1, lamd , add_matrix, sub_matrix, rev_matrix, del_matrix)
    
    stop = time.time()
    print('Time: ' + str(stop - start) + '\n')



    
    