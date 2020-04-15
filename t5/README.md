# Overview

This is my version of T5 for advice. T5 is a very big model. That means it only works on TPUs with model parallelism. That's also the reason for the [mesh tensorflow library](https://github.com/tensorflow/mesh), which contains much of the dirty work. I had to [fork it here](https://github.com/rowanz/mesh) to support nucleus sampling.

## Setup for training
You'll need a very large TPU pod, which you can spawn using a command like

```
ctpu up --tpu-size=v3-${numcores} --tf-version 1.15 --noconf --preemptible --tpu-only
```
The number of cores will depend on the model configuation (see `finetune.sh`). You can train it using `finetune.sh ${modelsize}`.  I don't know how to make this GPU compatible.

## Setup for validation and generation.

You'll need a (single) TPU v3 to use T5. Unfortunately, I don't know how to make this GPU compatible either. You can spawn one using a command like
`ctpu up --tpu-size=v3-8 --tf-version 1.15 --noconf --tpu-only`

You can get the perplexity (what I used for validation) using `validate.sh`. You can also run a server using `python run_server.py`

If you are going to use my checkpoints, please copy them first to your own google cloud storage account (so it doesn't keep charging mine!):

*   11B: `gs://turingadvice/baselines/t5/11B/model.ckpt-1010000.\*`
*    3B: `gs://turingadvice/baselines/t5/3B/model.ckpt-1018748.\*`
* large: `gs://turingadvice/baselines/t5/large/model.ckpt-1038196.\*`
*  base: `gs://turingadvice/baselines/t5/base/model.ckpt-1046772.\*`
* small: `gs://turingadvice/baselines/t5/small/model.ckpt-1037496.\*`

## NOTE about the demo

The demo right now is very inefficient. Essentially it seems like there's no way to keep a TPU model online for a while (like for a web demo). That means that, for each new API request you send the model, it'll have to initialize a new model, load all the parameters, and so on :sob:.

If you have any other (better) idea for how to implement the demo, please let me know!