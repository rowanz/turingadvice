# TuringAdvice

(aka, code for [Evaluating Machines by their Real-World Language Use](https://arxiv.org/abs/2004.03607).)

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
conda create --name turingadvice python=3.7 && conda activate news && conda install -y python=3.7 tqdm numpy pyyaml scipy ipython mkl mkl-include cython typing h5py pandas matplotlib lxml && pip install -r requirements.txt
```

Misc notes:
* I always have my pythonpath as the root directory. While in the `turingadvice` directory, run `export PYTHONPATH=$(pwd)` to set it.
* You might need to set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

### Bibtex

```
@article{zellers2020turingadvice,
    title={Evaluating Machines by their Real-World Language Use},
    author={Rowan Zellers and Ari Holtzman and Elizabeth Clark and Lianhui Qin and Ali Farhadi and Yejin Choi},
    journal={arXiv preprint},
    year={2020}
}
```