# data

Static dataset prep. Here, I have instructions and some scripts I used to create TFrecord files (which you need to train T5 and Grover).

# RedditAdvice2019.

You can download redditadvice2019 here:

`gsutil cp gs://turingadvice/redditadvice2019.jsonl .`

You can then use `to_tfrecord_grover.py` and `to_tfrecord_t5.py` to convert it into formats suitable for Grover and T5.





## Get all comments and posts from multiple subreddits using bigquery

The best way to download stuff from reddit is using BigQuery. See [https://pushshift.io/using-bigquery-with-reddit-data/](https://pushshift.io/using-bigquery-with-reddit-data/) for a walkthrough. Anyways, I used the following commands to create two tables in Google Cloud, one for posts and one for comments:
```
select created_utc,subreddit,author,num_comments,score,title,selftext,id,gilded,retrieved_on
from (select * from `fh-bigquery.reddit_posts.full_corpus_201512` union all select * from `fh-bigquery.reddit_posts.201*`)
where (subreddit = 'relationships' OR subreddit = 'Advice' OR subreddit = 'needadvice' OR subreddit = 'dating_advice' OR subreddit = 'dating' OR subreddit = 'love' OR subreddit = 'Marriage' OR subreddit = 'relationship_advice' OR subreddit = 'internetparents' OR subreddit = 'NoStupidQuestions' OR subreddit = 'techsupport' OR subreddit = 'legaladvice')
and score > 10
and char_length(selftext) > 64
and stickied = false;
```
and write to `posts` in `rowan_reddit_advice`. Then do
```
select created_utc,subreddit,author,parent_id,link_id,score,body,id,gilded,retrieved_on
from `fh-bigquery.reddit_comments.20*`
where (subreddit = 'relationships' OR subreddit = 'Advice' OR subreddit = 'needadvice' OR subreddit = 'dating_advice' OR subreddit = 'dating' OR subreddit = 'love' OR subreddit = 'Marriage' OR subreddit = 'relationship_advice' OR subreddit = 'internetparents' OR subreddit = 'NoStupidQuestions' OR subreddit = 'techsupport' OR subreddit = 'legaladvice')
and score > 10
and char_length(body) > 32;
```
Write this to `comments`.
