This repo uses reddit data from pushshift.io to create a chat bot. It's like the [huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai)  repo it was forked from, but the personalities are subreddits, the history are the previous comments, and utterances are from reddit comments. 

## Running

- clone the repo `git clone https://github.com/wassname/transfer-learning-conv-ai.git`
- install the requirements in `requirements.txt` you will need `pytorch-pretrained-bert>0.6.2` which may be the latest git release
- get data with `fetch_pushshift_data.py` (run overnight)
- with a nvidia 2080 ti you can run with `python -m ipdb train.py --dataset_path ./data/reddit_threads --fp16 O2 --gradient_accumulation_steps 32 --train_batch_size 1 --valid_batch_size 1 --n_epochs 3`
- interact with `python interact.py --device cpu --dataset_path ./data/reddit_threads --model_checkpoint runs/Jul09_15-44-47_mjcdesktop/`

If you want me to share the dataset and a pretrained model, just ask in the issues. I'm hesitant to do the work to share it if no one wants it.

## Current results

Are OK (see TODO below for ideas)

```
--------------------------------------------------------------------------------
2019-07-11 07:00:35 mjcdesktop interact.py[19521] INFO Selected personality: singularity
robot: This is the future, all these stupid people are going to take this seriously
human: Fake news right, it will be everywhere. We need to learn new norms as a society.
robot: It's not about ethics in journalism. It's about ethics in politics and the media.
human: And as consumers too.
robot: I hope that's not too late.
human: Well we're not dead. It took people a while to adjust to the printing press. We mainly consilidated newspapers into a few big ones the government could have a relationship with. So all aboard facebook & google I guess.
robot: I would say it's more like they're trying to make an issue of consumer vs media. 
human: Yeah I see what you mean, which one is at faul for the degrading quality of news.
robot: We don't need to fear the media. We need to be afraid of the people.
human: I'm agraid of everyone so it's all g.
--------------------------------------------------------------------------------
2019-07-11 06:58:37 mjcdesktop interact.py[19521] INFO Selected personality: /r/copypasta
human: Hi I'm a Navy Seal
robot: I'm a Marine. I've been a Marine for 11 years.
human: 11 years that's longer than I've been alive.
robot: You're a Marine too.
human: Great, :) we are all marines on this blessed day
robot: &gt;11 years that's longer than I've been alive.
--------------------------------------------------------------------------------
2019-07-11 07:10:53 mjcdesktop interact.py[19521] INFO Selected personality: totallynotrobots
robot: That's exactly what I thought. 
human: I THOUGHT YOU HUMAN
robot: I WAS SCREAMING FELLOW HUMAN
human: AH I SEE, WE HUMANS LIKE TO SCREAM
robot: WE HUMANS LIKE TO SHOUT!
human: YES WE DO, WITH OUR MOUTHS MADE OF MEAT
robot: HA HA HA HA HA HA HA HA HA HA HA HA HA HA
human: HA HA HA AH AH AAAAAAAA
robot: HA HA HA HA HA HA HA
human: AAHHHHHHHHHHhh
robot: HA HA HA HA HA HA HA
human: SCREANING IWTH MY HUMAN MOUTH
robot: IT'S ALL YOU NEED TO DO TO GET IT TO LOOK GOOD
human: /scream mode off
robot: I'M SCREAMING, HUMAN! I AM SCREAMING!
human: I am no longer screaming
robot: I WAS SCREAMING FELLOW HUMAN
human: Thanks you for the good human interaction fello human
robot: THAT IS WHAT I MEANT TO DO TO FELLOW HUMAN
human: As did I, other mammal
robot: I AM NOT SCREAMING FELLOW HUMAN
human: You are still screaming, you must push less air through your human lungs when you creat sounds
robot: You are still screaming, you must push more air through your human lungs
human: No, no less
robot: HEY HUMAN, I WAS SCREAMING FELLOW HUMAN. I WAS SCREAMING
human: And you still are, check your human logic loop, it may be circular
robot: I AM SCREAMING FELLOW HUMAN. I WAS SCREAMING FELLOW HUMAN.
human: Correct, thank you. We humans should always be correct
```

## TODO:

- [ ] Get better results
  - [x] Without username
  - [x] Remove [deleted]
  - [x] Remove short comments
  - [x] remove mod comments
  - [ ] Restrict to specific subreddit
  - [ ] Try GPT2 medium on colab (wait for pytorch+TPU)
- [ ] Data
  - [ ] Scrape to one big file, perhaps sqllite or jsonl
- [ ] Deploy: if good results,
- [ ]  interact with it on IRC


## Errors:

- `2019-07-10 02:35:49 ip-172-31-39-133 ignite.engine.engine.Engine[7647] ERROR Engine run is terminating due to exception: Creating MTGP constants failed. at /opt/conda/conda-bld/pytorch_1556653099582/work/aten/src/THC/THCTensorRandom.cu:33.  `
  - This means your GPU is full, reduce batch size or get a GPU with more ram
- `TypeError: __init__() got an unexpected keyword argument 'log_dir`
  - This is a tensorflowX vs ignite version problem, [see this github issue](https://github.com/pytorch/ignite/issues/534), and try new or old versions of these packages.

-----

Forked from

# 🦄 Building a State-of-the-Art Conversational AI with Transfer Learning

The present repo contains the code accompanying the blog post [🦄 How to build a State-of-the-Art Conversational AI with Transfer Learning](https://medium.com/@Thomwolf/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313).

This code is a clean and commented code base with training and testing scripts that can be used to train a dialog agent leveraging transfer Learning from an OpenAI GPT and GPT-2 Transformer language model.

This codebase can be used to reproduce the results of HuggingFace's participation to NeurIPS 2018 dialog competition [ConvAI2](http://convai.io/) which was state-of-the-art on the automatic metrics. The 3k+ lines of competition code was distilled in about 250 lines of training code with distributed & FP16 options to form the present repository.

This model can be trained in about one hour on a 8 V100 cloud instance (currently costs about $25) and a pre-trained model is also made available.

## Installation

To install and use the training and inference scripts please clone the repo and install the requirements:

```bash
git clone https://github.com/huggingface/transfer-learning-conv-ai
cd transfer-learning-conv-ai
pip install -r requirements.txt
```

## Installation with Docker

To install using docker please build the self-contained image:

```bash
docker build -t convai .
```

You can then enter the image  

```bash
ip-192-168-22-157:transfer-learning-conv-ai loretoparisi$ docker run --rm -it convai bash
root@91e241bb823e:/# ls
Dockerfile  README.md  boot                  dev  home         lib    media  models  proc              root  sbin  sys  train.py  utils.py
LICENCE     bin        convai_evaluation.py  etc  interact.py  lib64  mnt    opt     requirements.txt  run   srv   tmp  usr       var
```

You can then run the `interact.py` script on the pretrained model:

```bash
python3 interact.py --model models/
```

## Pretrained model

We make a pretrained and fine-tuned model available on our S3 [here](https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz). The easiest way to download and use this model is just to run the `interact.py` script to talk with the model. Without any argument, this script will automatically download and cache our model.

## Using the training script

The training script can be used in single GPU or multi GPU settings:

```bash
python ./train.py  # Single GPU training
python -m torch.distributed.launch --nproc_per_node=8 ./train.py  # Training on 8 GPUs
```

The training script accept several arguments to tweak the training:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset. If empty download from S3.
dataset_cache | `str` | `'./dataset_cache.bin'` | Path or url of the dataset cache
model | `str` | `"openai-gpt"` | Path, url or short name of the model
num_candidates | `int` | `2` | Number of candidates for training
max_history | `int` | `2` | Number of previous exchanges to keep in history
train_batch_size | `int` | `4` | Batch size for training
valid_batch_size | `int` | `4` | Batch size for validation
gradient_accumulation_steps | `int` | `8` | Accumulate gradients on several steps
lr | `float` | `6.25e-5` | Learning rate
lm_coef | `float` | `1.0` | LM loss coefficient
mc_coef | `float` | `1.0` | Multiple-choice loss coefficient
max_norm | `float` | `1.0` | Clipping gradient norm
n_epochs | `int` | `3` | Number of training epochs
personality_permutations | `int` | `1` | Number of permutations of personality sentences
device | `str` | `"cuda" if torch.cuda.is_available() else "cpu"` | Device (cuda or cpu)
fp16 | `str` | `""` | Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)
local_rank | `int` | `-1` | Local rank for distributed training (-1: not distributed)

Here is how to reproduce our results on a server with 8 V100 GPUs (adapt number of nodes and batch sizes to your configuration):

```bash
python -m torch.distributed.launch --nproc_per_node=8 ./train.py --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=2 --valid_batch_size=2
```

This model should give a Hits@1 over 79, perplexity of 20.5 and F1 of 16.5 using the convai2 evaluation script (see below).

These numbers are slightly lower than the number we obtained in the ConvAI2 competition. Here is what you can tweak to reach the same results:

- in the ConvAI2 competition we also used tweaked position emebddings so that the history of the dialog always start at with the same embeddings. This is easy to add with pytorch-pretrained-bert and should improve the hits@1 metric.
- in the ConvAI2 competition we used a beam search decoder. While the results are better in term of f1 metric, our feeling is that the human experience is les compelling with beam search versus the nucleus sampling detector which is provided in the present repository.

## Using the interaction script

The training script saves all the experiments and checkpoints in a sub-folder named with the timestamp of the experiment in the `./runs` folder of the repository base folder.

You can then use the interactive script to interact with the model simply by pointing to this folder.

Here is an example command line to run the interactive script:

```bash
python ./interact.py --model_checkpoint ./data/Apr17_13-31-38_thunder/  # run the interactive script with a training checkpoint
python ./interact.py  # run the interactive script with the finetuned model on our S3
```

The fine-tuned model will gives FINAL Hits@1: 0.715

The interactive script accept a few arguments to tweak the decoding algorithm:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset. If empty download from S3.
dataset_cache | `str` | `'./dataset_cache.bin'` | Path or url of the dataset cache
model | `str` | `"openai-gpt"` | Path, url or short name of the model
max_history | `int` | `2` | Number of previous utterances to keep in history
device | `str` | `cuda` if `torch.cuda.is_available()` else `cpu` | Device (cuda or cpu)
no_sample | action `store_true` | Set to use greedy decoding instead of sampling
max_length | `int` | `20` | Maximum length of the output utterances
min_length | `int` | `1` | Minimum length of the output utterances
seed | `int` | `42` | Seed
temperature | `int` | `0.7` | Sampling softmax temperature
top_k | `int` | `0` | Filter top-k tokens before sampling (`<=0`: no filtering)
top_p | `float` | `0.9` | Nucleus filtering (top-p) before sampling (`<=0.0`: no filtering)

## Running ConvAI2 evaluation scripts

To run the evaluation scripts of the ConvAI2 challenge, you first need to install `ParlAI` in the repo base folder like this:

```bash
git clone https://github.com/facebookresearch/ParlAI.git
cd ParlAI
python setup.py develop
```

You can then run the evaluation script from `ParlAI` base folder:

```bash
cd ParlAI
python ../convai_evaluation.py --eval_type hits@1  # to download and evaluate our fine-tuned model on hits@1 metric
python ../convai_evaluation.py --eval_type hits@1  --model_checkpoint ./data/Apr17_13-31-38_thunder/  # to evaluate a training checkpoint on hits@1 metric
```

The evaluation script accept a few arguments to select the evaluation metric and tweak the decoding algorithm:

Argument | Type | Default value | Description
---------|------|---------------|------------
eval_type | `str` | `"hits@1"` | Evaluate the model on `hits@1`, `ppl` or `f1` metric on the ConvAI2 validation dataset
model | `str` | `"openai-gpt"` | Path, url or short name of the model
max_history | `int` | `2` | Number of previous utterances to keep in history
device | `str` | `cuda` if `torch.cuda.is_available()` else `cpu` | Device (cuda or cpu)
no_sample | action `store_true` | Set to use greedy decoding instead of sampling
max_length | `int` | `20` | Maximum length of the output utterances
min_length | `int` | `1` | Minimum length of the output utterances
seed | `int` | `42` | Seed
temperature | `int` | `0.7` | Sampling softmax temperature
top_k | `int` | `0` | Filter top-k tokens before sampling (`<=0`: no filtering)
top_p | `float` | `0.9` | Nucleus filtering (top-p) before sampling (`<=0.0`: no filtering)

## Citation

If you use this code in your research, you can cite our NeurIPS CAI workshop [paper](http://arxiv.org/abs/1901.08149):

```bash
@article{DBLP:journals/corr/abs-1901-08149,
  author    = {Thomas Wolf and
               Victor Sanh and
               Julien Chaumond and
               Clement Delangue},
  title     = {TransferTransfo: {A} Transfer Learning Approach for Neural Network
               Based Conversational Agents},
  journal   = {CoRR},
  volume    = {abs/1901.08149},
  year      = {2019},
  url       = {http://arxiv.org/abs/1901.08149},
  archivePrefix = {arXiv},
  eprint    = {1901.08149},
  timestamp = {Sat, 02 Feb 2019 16:56:00 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1901-08149},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
