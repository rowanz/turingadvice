# grover

Grover baseline for TuringAdvice. You can train/validate the model in `lm`, or set up a server for generation in `server`.

The main command you'll probably want is under `server`: just use `python run_server.py` and you'll have a web demo working. You'll need to download checkpoints first, using `get_checkpoint.sh {modeltype}` where modeltype is (`base`, `large`, or `mega`).


# Checkpoints

You'll need to use google cloud storage for these.

Grover-{base,large,mega}, adapted for length up to 1536 BPE tokens but NOT finetuned on RedditAdvice2019:
```
gs://turingadvice/baselines/grover/grover_realnews_longsecondarypretrain_jan_1_2020/model={base,large,mega}~lr=2e-5~epochs=3/model.ckpt-18000.{index,meta,data-00000-of-00001}
```

Grover-Base (ppl=17.565): 
`gs://turingadvice/baselines/grover/base/model.ckpt-23436.\*`

Grover-Large (ppl=14.733): 
`gs://turingadvice/baselines/grover/large/model.ckpt-23436.\*`

Grover-Mega (ppl=12.56):
`gs://turingadvice/baselines/grover/mega/model.ckpt-23436.\*`

(For the finetuned checkpoints, you can also use [server/get_ckpt.sh])