#! Python3 -- applies ml to kaggle horror author dataset to identify authorship of unknown sentences. Useful for wider 18c project...

import numpy as np 
import pandas as pd 
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

sid = SentimentIntensityAnalyzer()

df = pd.read_csv('~/documents/github/ml-horror/train.csv')
#print(df.head(5)) if you want to check it's inputted correctly

# removes punctuation for word counting purposes.
no_punct_translator = str.maketrans('','', string.punctuation)
# tokenize each sentence and remove punctuation
df['words'] = df['text'].apply(lambda t: nltk.word_tokenize(t.translate(no_punct_translator).lower()))


# create a new column with the count of all words
df['word_count'] = df['words'].apply(lambda words: len(words))
# for normalization, how many characters per sentence w/o punctuation
df['sentence_length'] = df['words'].apply(lambda w: sum(map(len, w)))
# for future calculations, let's keep around the full text length, including punctuation
df['text_length'] = df['text'].apply(lambda t: len(t))

#print(df.head(5)) #checks everything's working correctly: it is!


sns.set_style('whitegrid') # sets style for graphs imported above

""" 
This exports the snsplot to a png in the folder.

sns_plot = sns.boxplot(x = "author", y = "word_count", data=df, color = "red") # plots word per sentence.
fig = sns_plot.get_figure()
fig.savefig('plot.png')
"""
#sns.boxplot(x = "author", y = "word_count", data=df, color = "red") # plots word per sentence.

""" prints chars in each sentence
print(df.groupby(['author'])['sentence_length'].describe())
"""

def unique_words(words):
    word_count = len(words)
    unique_count = len(set(words)) # creating a set from the list 'words' removes duplicates
    return unique_count / word_count

df['unique_ratio'] = df['words'].apply(unique_words)
# print(df.groupby(['author'])['unique_ratio'].describe())

# graphs unique word per author distribution.

authors = ['MWS', 'HPL', 'EAP']
"""
for author in authors:
   sns.distplot(df[df['author'] == author]['unique_ratio'], label = author, hist=False)

plt.legend();
"""
"""
sns_dist = sns.distplot(df[df['author'] == author]['unique_ratio'], label = author, hist=False)

fig2 = sns_dist.get_figure()
fig2.savefig('dist.png')

"""
""" prints sentiment analysis for these two example sentences 
print(sid.polarity_scores('Vader text analysis is my favorite thing ever'))
print(sid.polarity_scores('I hate vader and everything it stands for'))
""" 

df['sentiment'] = df['text'].apply(lambda t: sid.polarity_scores(t)['compound'])
df.groupby('author')['sentiment'].describe()


for author in authors:
    sns.distplot(df[df['author'] == author]['sentiment'], label = author, hist=False)

plt.legend();

sns_sent = sns.distplot(df[df['author'] == author]['sentiment'], label = author, hist=False)

fig3 = sns_sent.get_figure()
fig3.savefig('sentiment.png')

# makes a boxplot sns.boxplot(x="author", y="sentiment", data=df)

