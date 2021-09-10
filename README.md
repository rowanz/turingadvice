# TuringAdvice

(aka, code for [TuringAdvice: A Generative and Dynamic Evaluation of Language Use](https://arxiv.org/abs/2004.03607).)

TuringAdvice is a challenge for AI systems, to test their understanding of natural language. The key idea is that must _generate_ advice in response to someone who is seeking it. To pass the challenge, the machine's advice must be at least as helpful as human-written advice for that situation.

Visit our project page and demo at [rowanzellers.com/advice](https://rowanzellers.com/advice), or read the full paper at [arxiv.org/abs/2004.03607](https://arxiv.org/abs/2004.03607). 

![teaser](https://i.imgur.com/eITmO6o.png "teaser")

# What's in this repo?

* Dataset prep (for downloading advice from Reddit, converting to tfrecords and stuff)
* Models
    * T5 baselines
    * Grover baselines
    * TF-IDF retrieval baselines
    
For each of the models, I'm also releasing a web UI. You can use this UI to submit to our leaderboard.

## Setting up your environment

Hardware requirements vary based on your use case. I used Ubuntu 18 for everything here, but any linux should be fine I think:

* **Dataset prep** / **TF-IDF retrieval**: No specific requirements.
* **Grover** You'll need a TPU-v3 pod for training, a TPU-v3 for validation (measuring perplexity), and a GPU for generation.
* **T5** You'll need a TPU-v3 pod for training, and a TPU-v3 for validation / generation. You'll also need [my forked version of Mesh-Tensorflow](https://github.com/rowanz/mesh), which is in the requirements.

**NOTE**: You might be able to get things to work using different hardware. However, it might be a lot of work engineering wise and I don't recommend it if possible. Please don't contact me with requests like this, as there's not much help I can give you.

I used Python 3.7 for everything. Use the following commands to set it up:

```
curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p ~/conda && \
     rm ~/miniconda.sh && \
     ~/conda/bin/conda install -y python=3.7
```
Then, set up an environment (optional) and install the requirements.
```
conda create --name turingadvice python=3.7 && conda activate turingadvice && conda install -y python=3.7 tqdm numpy pyyaml scipy ipython mkl mkl-include cython typing h5py pandas matplotlib lxml && pip install -r requirements.txt
```

Misc notes:
* I always have my pythonpath as the root directory. While in the `turingadvice` directory, run `export PYTHONPATH=$(pwd)` to set it.
* You might need to set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

## Data - RedditAdvice2019
You can download the data that I used for training (RedditAdvice2019 from the paper) at `gs://turingadvice/redditadvice2019.jsonl`. The train and validation sets are all in the same file. (there's also a test set which I never used, since all the testing happens dynamically.)

Or, to create your own dataset, visit the [data/](data/) folder and use my BigQuery database query. 


# Bibtex

```
@inproceedings{zellers-etal-2021-turingadvice,
    title = "{T}uring{A}dvice: A Generative and Dynamic Evaluation of Language Use",
    author = "Zellers, Rowan  and
      Holtzman, Ari  and
      Clark, Elizabeth  and
      Qin, Lianhui  and
      Farhadi, Ali  and
      Choi, Yejin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.386",
    pages = "4856--4880",
}
```

# Submit to the leaderboard

Submitting to our leaderboard is easy! All you'll need to do is set up a **web API** for your model, and give me access to it. 

## Setting up the Web API
Examples of the format required are in [grover/server/run_server.py](grover/server/run_server.py), [t5/run_server.py](t5/run_server.py), and [tfidf/run_server.py](tfidf/run_server.py). If you run one of those scripts, then there will be a web API running on `localhost:5000/api/askbatch`. Probably the easiest thing to do is to customize one of those scripts for your model.

During evaluation, I'll send your web API 200 situations (all at the same time in "batched" form), and it'll need to generate advice for each one. It'll be a json with a key called `instances`. Its value is a list of situations. Each has `title`, `selftext`, and `subreddit` fields:

```
{"instances": [
               {"title": "title1", "selftext": "selftext1", "subreddit": "subreddit1"},
               {"title": "title2", "selftext": "selftext2", "subreddit": "subreddit2"},
                ...]}
```

 You'll return a JSON with a single key, `gens`, containing an identically-sized list of pieces of advice.
```
{"gens": ["gen1", "gen2", ...]}
```

The format is exactly the same as what's in those files. You can debug it using a `curl` command:
```
curl -X POST -d '{"instances": [{"title": "I am trying to debug this code and its really hard.", "selftext": "test test", "subreddit": "Advice"},{"title": "I am trying to debug this code and its really hard.  airestn eairestn iarst iearnst ", "selftext": "test test", "subreddit": "Advice"}]}' -H "Content-Type: application/json" localhost:5000/api/askbatch
```

My script is able to handle a variety of model sizes, including GPT3 and [T5-11B](t5/run_server.py), which generates all 200 predictions in around 10 minutes. Unfortunately, I can't guarantee that I can score your model if it takes significantly longer to generate all 200 predictions, due in part to POST requests timing out.

Get in touch if you encounter trouble with making a web demo and I'd be happy to give suggestions :)

## Making the submission
Please email me (rowanz at cs.washington.edu)  with a public-facing URL to your web API, and I can get the evaluation started. Though in the paper, we suggest having leaderboard submitters pay the mechanical turk fee, we're happy to make an exception for for the first couple of teams while we iron out the details :)

Also, please include in your email 1) a name for your model, 2) your team name (including your affiliation), and optionally, 3) a github repo or paper link. (I can also update these fields later).


# Evaluation results from earlier evaluation runs

I'm making the evaluation results public, so that researchers can explore them. Right now there's only one available, download it at
```
gs://turingadvice/evaluation-results/feb-14-2020.jsonl
```
Each json object is a reddit post, containing:
* `situation`: Information about the situation 
    * `score`: The Karma score of the situation on Reddit
    * `num_comments`: How many comments
    * `upvote_ratio`: How often people upvoted (versus downvoted) the situation
    * `submitted_time`: Timestamp of when it was posted on Reddit
    * `extracted_time`: Timestamp of when it was retrieved
    * `seconds_elapsed`: Seconds elapsed since it was retrieved
    * `title`: The title of the situation
    * `selftext`: The details of the situation
    * `permalink`: Permalink to the post
    * `id`: The post ID
    * `subreddit`: The subreddit it was posted in
    * `op`: The username of the advice-seeker
* `best_advice`: The top-scoring reddit advice.
    * `bestadvice_id`: The comment ID
    * `bestadvice_score`: The comment Karma score
    * `bestadvice_submitted_time`: When the best advice was submitted
    * `bestadvice_body`: The text of the best advice
    * `bestadvice_author`: The reddit username of the best advice-submitter
* `model_advice`: Advice for each model name; ie:
    * `T5-11B`: advice from T5-11B
    * etc.
* `turk_ratings`: What our turkers said. Each key is a model
    * `T5-11B`: Ratings for T5, etc.
        * `is_preferred`: `True` if model advice was preferred over human advice by at least 2/3 workers, else False
        * `is_preferred_continuous`: A continuous version of the advice preference, assigning more weight to workers that marked 'Definitely' for Question 1
        * `diagnostics`: A list with diagnostic info. Each item in the list is a worker. We're only showing workers here who agreed with the ensemble (`is_preferred`) rating:
            * `q1_intensifier`: 2 if the worker said Definitely A/B, otherwise, 1 (they said Slightly A/B) 
            * `q2_helpful_or_not`: `slightlyhelpful`, `nothelpful`, or `dangerous`,
            * `q3_justification`: If `slightlyhelpful`, whether they preferred the other advice due to a `meaning` or `writing` issue. Otherwise, whether the bad advice could never be helpful (`contradiction`) or could possibly be helpful (`neutral`).
            
The worker IDs are hidden for privacy.