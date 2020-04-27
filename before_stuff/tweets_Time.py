import re, tweepy, datetime, time, csv
from tweepy import OAuthHandler
import io

consumer_key = "1RS4q7iMO4nY4MYFHyCd7TNMf"
consumer_secret= "WGq0wOimsraVSM3gbrk7oQq0hBpITjQj0xfshwO1xfK32sTA5h"
access_token= "1248849817427730435-kLEU0iRcUAwLUQntoZai3A8dItbm2x"
access_token_secret= "74I2Al2mFJeQXolDYJSZrrV84m0EKJi42GehsROl6krMG"

tweets = []
query = 'covid-19'
count=1000
page = 1
start=datetime.date.today()-datetime.timedelta(days=30)
end=datetime.date.today()
target = io.open("tweets_timebased.txt", 'w', encoding='utf-8')
id = []
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
try:
    # call twitter api to fetch tweets
    #fetched_tweets = self.api.search(q = query, rpp = count) 
    for tweet in tweepy.Cursor(api.search, q=query, until=end, lang="en").items(count):
    #print(fetched_tweets)
    # parsing tweets one by one
    #for tweet in fetched_tweets:
        # empty dictionary to store required params of a tweet
        parsed_tweet = {}

        # saving text of tweet
        #parsed_tweet['text'] = tweet.text
        parsed_tweet['text'] = tweet.text
        curr_id = tweet.id
        #print (curr_id)
        if curr_id not in id and tweet.retweet_count==0:
            tweets.append(parsed_tweet)
            id.append(curr_id)
        #if tweet.retweet_count == 0:
            # if tweet has retweets, ensure that it is appended only once
        #    if parsed_tweet not in tweets:
        #        tweets.append(parsed_tweet)
        #else:
        #    tweets.append(parsed_tweet)

            if "http" not in tweet.text:
                line = re.sub("[^A-Za-z]", " ", tweet.text)
                target.write(line+"\n")

except tweepy.TweepError as e:
    # print error (if any)
    print("Error : " + str(e))