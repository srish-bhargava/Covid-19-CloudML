REGISTER piggybank-0.12.0.jar;

define CSVLoader org.apache.pig.piggybank.storage.CSVLoader();

users = LOAD 'users.csv' USING org.apache.pig.piggybank.storage.CSVLoader() 
	AS (userlogin:chararray, username:chararray, state:chararray);

tweets = LOAD 'tweets.csv' USING org.apache.pig.piggybank.storage.CSVLoader() 
	AS (tweetid:chararray, tweetcontent:chararray, userlogin:chararray);

A = FILTER tweets BY (tweetcontent matches '.*favorite.*');
DUMP A;
B = ORDER A BY tweetid;
C = FOREACH B GENERATE tweetid, tweetcontent;
STORE C into '1b.results' USING PigStorage ();
