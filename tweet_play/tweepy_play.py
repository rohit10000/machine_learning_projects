import tweepy
from textblob import TextBlob
import pandas as pd
import numpy as np
consumer_key = 'Xqb4Qbaw3XyYPZcuse6RckE4R'
consumer_secret = 'sa7ulPMnJUqDwhTQcrOvv309NAmo1cy9cHbee2AuJ5wThK9Sww'
access_token  = '783315261432528897-C05MZwp1xrpmYasxJS0i3yg6XpHAlUw'
access_token_secret = 'ZgylLwHaai2taqlsAn4DJNISFAkqEtlcPBSoEweLsOwWt'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)
public_tweets = api.search('Modi')
tweets = []
polarity = []
subjectivity = []

for tweet in public_tweets:
	analysis = TextBlob(tweet.text)
	tweets.append(tweet.text.encode('utf-8'))
	polarity.append(analysis.polarity)
	subjectivity.append(analysis.subjectivity)

list_labels = ['tweet_text','polarity','subjectivity']
list_cols = [tweets, polarity, subjectivity]

zipped = zip(list_labels,list_cols)
#print zipped
data = dict(zipped)
dataset = pd.DataFrame(data)

filename = 'labeled_tweet.csv'
dataset.to_csv(filename, encoding='utf-8', index=False)