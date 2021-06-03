'''
Dependencies:
1. pandas
2. sklearn
3. nltk: 
    https://www.nltk.org/install.html for installing nltk
    https://www.nltk.org/data.html for install nltk data, stopwords need to be downloaded here
4. textblob:
    https://textblob.readthedocs.io/en/dev/install.html for installing textblob and its data

How to run:
Modify the global variable 'output_location' in the first section of the code to change the location of the output file, then run the python script 
'''

import pandas as pd
import sklearn
import datetime
from ast import literal_eval
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

#############################################################Read data################################################################
train_x = pd.read_csv('./data/p_train_x.csv', sep=',', 
    dtype={"id": 'int64',"conversation_id": 'int64','user_id': 'int64', 'video':'int', 'name':'string'})
train_y = pd.read_csv('./data/p_train_y.csv', sep=',')
test_x = pd.read_csv('./data/p_test_x.csv', sep=',', 
    dtype={"id": 'int64',"conversation_id": 'int64','user_id': 'int64', 'video':'int', 'name':'string'})

output_location = r'D:\cs480_project\data\prediction.csv'
#############################################################Data processing and feature extraction################################################################
# Helper functions
# Get average word length in a tweet
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

stop_words=set(stopwords.words("english"))
def remove_stop_words(sentence):
    filtered_sent=[]
    for w in sentence:
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent

# Convert create_at to ordinal datetime
train_x['created_at'] = train_x['created_at'].apply(lambda x: datetime.datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S').toordinal())
test_x['created_at'] = test_x['created_at'].apply(lambda x: datetime.datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S').toordinal())
# Word count in a tweet
train_x['tweet_word_count'] = train_x['tweet'].apply(lambda x: len(word_tokenize(x)))
test_x['tweet_word_count'] = test_x['tweet'].apply(lambda x: len(word_tokenize(x)))
# Char count in a tweet
train_x['tweet_char_count'] = train_x['tweet'].apply(lambda x: len(x))
test_x['tweet_char_count'] = test_x['tweet'].apply(lambda x: len(x))
# Number of upper case words in a tweet
train_x['tweet_num_upper_word'] = train_x['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
test_x['tweet_num_upper_word'] = test_x['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
# Average length in a tweet
train_x['tweet_avg_word_len'] = train_x['tweet'].apply(avg_word)
test_x['tweet_avg_word_len'] = test_x['tweet'].apply(avg_word)
# Numeric count in a tweet
train_x['tweet_num_count'] = train_x['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
test_x['tweet_num_count'] = test_x['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
# Word count in tweet without stopwords
train_x['cleaned_tweet_num_count'] = train_x['tweet'].apply(remove_stop_words).apply(lambda x: len([x for x in x if x.isdigit()]))
test_x['cleaned_tweet_num_count'] = test_x['tweet'].apply(remove_stop_words).apply(lambda x: len([x for x in x if x.isdigit()]))
# Average length in a tweet without stopwords
train_x['cleaned_tweet_avg_word_len'] = train_x['tweet'].apply(remove_stop_words).apply(lambda words: (sum(len(word) for word in words)/len(words)))
test_x['cleaned_tweet_avg_word_len'] = test_x['tweet'].apply(remove_stop_words).apply(lambda words: (sum(len(word) for word in words)/len(words)))
# Tweet sentiment polarity
train_x['tweet_polarity'] = train_x['tweet'].apply(lambda x: TextBlob(x).sentiment[0])
test_x['tweet_polarity'] = test_x['tweet'].apply(lambda x: TextBlob(x).sentiment[0])
# Tweet sentiment subjectivity
train_x['tweet_subjectivity'] = train_x['tweet'].apply(lambda x: TextBlob(x).sentiment[1])
test_x['tweet_subjectivity'] = test_x['tweet'].apply(lambda x: TextBlob(x).sentiment[1])
# Whether a tweet is in English, 0 for no, 1 for yes
train_x['is_english'] = train_x['language'].apply(lambda x: 1 if x == 'en' else 0)
test_x['is_english'] = test_x['language'].apply(lambda x: 1 if x == 'en'else 0)
# Whether a tweet has location, 0 for no, 1 for yes
train_x['has_place'] = train_x['place'].apply(lambda x: 0 if pd.isnull(x) else 1)
test_x['has_place'] = test_x['place'].apply(lambda x: 0 if pd.isnull(x) else 1)
# Mention count
train_x['mention_count'] = train_x['mentions'].apply(lambda x: len(literal_eval(x)))
test_x['mention_count'] = test_x['mentions'].apply(lambda x: len(literal_eval(x)))
# URL count
train_x['url_count'] = train_x['urls'].apply(lambda x: len(literal_eval(x)))
test_x['url_count'] = test_x['urls'].apply(lambda x: len(literal_eval(x)))
# Photo count
train_x['photo_count'] = train_x['photos'].apply(lambda x: len(literal_eval(x)))
test_x['photo_count'] = test_x['photos'].apply(lambda x: len(literal_eval(x)))
# Cashtag count
train_x['cashtag_count'] = train_x['cashtags'].apply(lambda x: len(literal_eval(x)))
test_x['cashtag_count'] = test_x['cashtags'].apply(lambda x: len(literal_eval(x)))
# Whether tweet has quote url, 0 for no, 1 for yes
train_x['has_quote_url'] = train_x['quote_url'].apply(lambda x: 0 if pd.isnull(x) else 1)
test_x['has_quote_url'] = test_x['quote_url'].apply(lambda x: 0 if pd.isnull(x) else 1)
# Number of replies
train_x['reply_to_count'] = train_x['reply_to'].apply(lambda x: len(literal_eval(x)))
test_x['reply_to_count'] = test_x['reply_to'].apply(lambda x: len(literal_eval(x)))
# Name words count
train_x['name_words_count'] = train_x['name'].apply(lambda x: len(str(x).split(" ")))
test_x['name_words_count'] = test_x['name'].apply(lambda x: len(str(x).split(" ")))
# Char count in Name
train_x['name_char_count'] = train_x['name'].apply(lambda x: len(str(x)))
test_x['name_char_count'] = test_x['name'].apply(lambda x: len(str(x)))
# Retain only numeric fields, remove id 
train_x = train_x.select_dtypes(include=['number'])
test_x = test_x.select_dtypes(include=['number'])
train_x = train_x.drop(columns=['id', 'conversation_id', 'user_id'])
test_x = test_x.drop(columns=['id', 'conversation_id', 'user_id'])
#############################################################Train Model#######################################################################################
# Train a random forest model with 6000 trees
model = RandomForestClassifier(n_estimators=5000, 
                               bootstrap = True,
                               max_features = 'sqrt')
model.fit(train_x, train_y['likes_count'])
#############################################################Produce predicts#######################################################################################
predictions = model.predict(test_x)
prediction = pd.DataFrame(predictions, columns=['label']).to_csv(output_location, index = True, index_label='id', header=True)