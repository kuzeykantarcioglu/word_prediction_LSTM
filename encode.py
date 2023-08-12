import json
from math import random 
def create_data(length):
    seq=[]
    start=random.randint(0, len(cleaned_corpus)//2) # Used half of the corpus due to memory error
    for i in range(length,len(cleaned_corpus)//2):
        words = cleaned_corpus[i-length+start:i+start]
        line = ' '.join(words)
        seq.append(line)
        if i % 2000000==0:
            print(i , 'tokens done')
    with open('len'+str(length)+'.json', 'w') as fp:
        json.dump(seq, fp)
create_data(2)
create_data(4)
create_data(7)


def encoding_data(length):
    with open('len'+str(length)+'.json', 'r') as fp:
        seq=json.load( fp)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(seq[:200000])
    
    sequences = tokenizer.texts_to_sequences(seq[:200000])
    
    sequences=np.array(sequences)
    vocab=len(tokenizer.word_counts)+1
    data_x=sequences[:,:-1]
    data_y=sequences[:,-1]
    data_y = to_categorical(data_y, num_classes=vocab)
    words_to_index = tokenizer.word_index
    with open('tokenizer_len'+str(length)+'.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del seq
    return data_x,data_y,vocab,words_to_index