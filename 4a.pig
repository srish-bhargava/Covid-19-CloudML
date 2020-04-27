REGISTER piggybank-0.12.0.jar;

define CSVLoader org.apache.pig.piggybank.storage.CSVLoader();

users = LOAD 'users.csv' USING org.apache.pig.piggybank.storage.CSVLoader() 
	AS (userlogin:chararray, username:chararray, state:chararray);

tweets = LOAD 'tweets.csv' USING org.apache.pig.piggybank.storage.CSVLoader() 
	AS (tweetid:chararray, tweetcontent:chararray, userlogin:chararray);

A = JOIN users BY userlogin, tweets BY userlogin;
B = FOREACH A GENERATE users::userlogin, state, username, tweetcontent, tweetid;
C = GROUP B BY username;
D = FOREACH C GENERATE group, COUNT_STAR(B.tweetcontent) AS tweet_count;
E = ORDER D BY tweet_count DESC;
F = FILTER E BY tweet_count >=2;
G = FOREACH F GENERATE group;
STORE G into '4a.results' USING PigStorage ();



