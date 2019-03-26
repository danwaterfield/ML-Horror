#! Python3 -- applies ml to kaggle horror author dataset to identify authorship of unknown sentences. Useful for wider 18c project...

import numpy as np 
import pandas as pd 
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from wordcloud import WordCloud, STOPWORDS
import tensorflow as tf

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
avg_length = lambda words: sum(map(len, words)) / len(words)

df['avg_word_length'] = df['words'].apply(avg_length)
df.groupby(['author'])['avg_word_length'].describe()
df.groupby(['author'])['sentence_length'].describe()

df['punctuation_count'] = df['text'].apply(lambda t: len(list(filter(lambda c: c in t, string.punctuation))))

df['punctuation_per_char'] = df['punctuation_count'] / df['text_length'] 


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





# iterate all rows and create a new dataframe with author->word (single word)
df_words = pd.concat([pd.DataFrame(data={'author': [row['author'] for _ in row['words']], 'word': row['words']})
           for _, row in df.iterrows()], ignore_index=True)

# use NLTK to remove all rows with simple stop words
df_words = df_words[~df_words['word'].isin(nltk.corpus.stopwords.words('english'))]

df_words.shape


def authorWordcloud(author):
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40,background_color="black", max_words=10000).generate(" ".join(df_words[df_words['author'] == author]['word'].values))
    plt.figure(figsize=(11,13))
    plt.title(author, fontsize=16)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

"""again, to save as an image file you need to assign each wordcloud to a variable, then use safefig('name.png etc here)
    
authorWordcloud('HPL')
authorWordcloud('EAP')
authorWordcloud('MWS')
"""

# function for a specific author to count occurances of each word
def authorCommonWords(author, numWords):
    authorWords = df_words[df_words['author'] == author].groupby('word').size().reset_index().rename(columns={0:'count'})
    authorWords.sort_values('count', inplace=True)
    return authorWords[-numWords:]

# for example, here's how we get the 10 most common EAP words.
# print(authorCommonWords('EAP', 10)) 





authors_top_words = []
for author in authors:
    authors_top_words.extend(authorCommonWords(author, 10)['word'].values)

# use a set to remove duplicates
authors_top_words = list(set(authors_top_words))

df['top_words'] = df['words'].apply(lambda w: list(set(filter(set(w).__contains__, authors_top_words))))
df[['author','top_words', 'words']].head()


feature_columns = ['author', 'word_count', 'text_length', 'punctuation_per_char', 'unique_ratio', 'avg_word_length', 'sentiment']
df_features = df[feature_columns]


df_train=df_features.sample(frac=0.8,random_state=1)
df_dev=df_features.drop(df_train.index)

print(df_train.head())



