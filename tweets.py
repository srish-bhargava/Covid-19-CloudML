from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

consumer_key = "1RS4q7iMO4nY4MYFHyCd7TNMf"
consumer_secret= "WGq0wOimsraVSM3gbrk7oQq0hBpITjQj0xfshwO1xfK32sTA5h"
access_token= "1248849817427730435-kLEU0iRcUAwLUQntoZai3A8dItbm2x"
access_token_secret= "74I2Al2mFJeQXolDYJSZrrV84m0EKJi42GehsROl6krMG"

class StdOutListener(StreamListener):
    
    def on_data(self, data):
        print (data)
        return True

    def on_error(self, status):
        print (status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(track=['covid-19', 'quarantine', 'lockdown'])