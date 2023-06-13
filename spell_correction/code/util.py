import ast

ADD_MATRIX = '../confusion_matrix/addconfusion.data'
DEL_MATRIX = '../confusion_matrix/delconfusion.data'
REV_MATRIX = '../confusion_matrix/revconfusion.data'
SUB_MATRIX = '../confusion_matrix/subconfusion.data'


def edit_finder(candidate , word):
    """
    candidate: the word most likely to be correct
    word: the origin wrong word

    method to calculate edit type for single edit errors
    """
    if word == candidate:
        return '' , '' , '' , '' , ''

    wrong_reason = ''
    correct = ''
    error = ''
    x = ''
    w = ''
    for i in range(max(len(word) , len(candidate))):
        if candidate[0:i +1] != word[0:i +1]: # find the first difference in i
            # letter in i was delete from candidate
            if candidate[i+1 :] == word[i :]:
                correct = candidate[i]
                error = ''
                x = candidate[i-1] # 待修改的样子
                w = candidate[i-1] + candidate[i] # 修改后的样子
                wrong_reason = 'Deletion'
                break

            # letter was inserted to candidate
            if candidate[i:] == word[i+1 :]:
                correct = ''
                error = word[i]
                if i == 0:
                    w = '#'
                    x = '#' + error
                else:
                    w = word[i-1]
                    x = word[i-1] + error
                wrong_reason = 'Insertion'
                break
                
            # letter was exchanged from candidate
            if candidate[i+1 :] == word[i+1 :]:
                correct = candidate[i]
                error = word[i]
                x = error
                w = correct
                wrong_reason = 'Substitution'
                break

            # two letters was reversed from candidate
            if candidate[i+1] == word[i] and candidate[i] == word[i+1] and candidate[i+2 :] == word[i+2 :]:
                correct = candidate[i] + candidate[i+1]
                error = word[i] + word[i+1]
                x = error
                w = correct
                wrong_reason = 'Reversal'
                break

    return wrong_reason , correct , error , x , w


def load_confusion_matrix():
    f = open(ADD_MATRIX , 'r')
    data = f.read()
    f.close
    add_matrix = ast.literal_eval(data) # safer than 'eval()'

    f = open(SUB_MATRIX , 'r')
    data = f.read()
    f.close
    sub_matrix = ast.literal_eval(data)

    f = open(REV_MATRIX , 'r')
    data = f.read()
    f.close
    rev_matrix = ast.literal_eval(data)

    f = open(DEL_MATRIX , 'r')
    data = f.read()
    f.close
    del_matrix = ast.literal_eval(data)
    
    return add_matrix , sub_matrix , rev_matrix , del_matrix
        
if __name__ == '__main__':
    print(edit_finder('barely' , 'barels'))
    print(edit_finder('could' , 'coul'))
    print(edit_finder('mine' , 'miney'))
    print(edit_finder('revenues' , 'ervenues'))