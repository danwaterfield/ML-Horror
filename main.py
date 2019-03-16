#! Python3 -- applies ml to kaggle horror author dataset to identify authorship of unknown sentences. Useful for wider 18c project...


import numpy as np 
import pandas as pd 
import nltk 
import string

df = pd.read_csv('~/documents/github/ml-horror/train.csv')
# removes punctuation for word counting purposes.
no_punct_translator=str.maketrans('','',string.punctuation)
# tokenize each sentence and remove punctuation
df['words'] = df['text'].apply(lambda t: nltk.word_tokenize(t.translate(no_punct_translator).lower()))


# create a new column with the count of all words
df['word_count'] = df['words'].apply(lambda words: len(words))
# for normalization, how many characters per sentence w/o punctuation
df['sentence_length'] = df['words'].apply(lambda w: sum(map(len, w)))
# for future calculations, let's keep around the full text length, including punctuation
df['text_length'] = df['text'].apply(lambda t: len(t))

print(df.head(5))