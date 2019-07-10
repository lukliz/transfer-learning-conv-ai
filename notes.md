- get data with `fetch_pushshift_data.py` (run overnight)
- concat with `2_`

run
`python ./train.py --dataset_path ./data/reddit_threads/totallynotrobots_train.text --dataset_cache .cache/dataset_cache.bin`

TODO:
- [ ] Remove personality
- [ ] Load out data
- [ ] format with special tokens see SPECIAL_TOKENS in train.py

Hmm how to format data? Easiest way is to have
- for each comment: we have a submission=persona, parents=history, candiates are child comments. I can look at author to get person1 or person2



# Loading data

I thought about formatting it using special  chars etc, but I decide it would be easier to use hugging faces layout rather than rewrite their whole repo. So for each reddit comment, it has a history, the submission is the personality, and pssible replies are candiates.

- [x] Loaded comments into these formats
- [x] make them match the json format
- [x] try to load
- [x] refactor and load in term
- [x] fix cuda, gpu errors
- [x] fix dataset size lim
    - [x]  TODO this is currently giving a max of 1000
    - [x]  fix data download limt
    - [x]  fix dataset size
    - [x]  fix the seq length, it seems that I need to concat when adding hist etc
- [x] change formatting
  - [x] remove the ids, we don't need it in this format
- [x] interact
- [x] remove [remove] and [deleted] as well as mod flaired messaged
wishlist: save in json not pickle., append to jsonl, make the download script work better by getting a wekek at a time

Look like I can't fit it into mem, need to use a cloud gpu!
`python train.py --dataset_path ./data/reddit_threads --fp16 O3 --gradient_accumulation_steps 8 --train_batch_size 1 --valid_batch_size 1`

works!
- O2 causes nans
`python train.py --dataset_path ./data/reddit_threads --fp16 O1 --gradient_accumulation_steps 8 --train_batch_size 2 --valid_batch_size 2`


except:
INFO:train.py:Train dataset (Batch, Candidates, Seq length): torch.Size([8, 2, 279])
INFO:train.py:Valid dataset (Batch, Candidates, Seq length): torch.Size([2, 3, 199])

`python train.py --dataset_path ./data/reddit_threads --fp16 O1 --gradient_accumulation_steps 16 --train_batch_size 2 --valid_batch_size 2 --n_epochs 10`
`python train.py --dataset_path ./data/reddit_threads --fp16 O2 --gradient_accumulation_steps 32 --train_batch_size 1 --valid_batch_size 1 --n_epochs 10`

`python interact.py --device cpu --dataset_path ./data/reddit_threads --model_checkpoint runs/Jul09_15-44-47_mjcdesktop/`

python interact.py --dataset_path ./data/reddit_threads --model_checkpoint runs/Jul09_21-25-16_mjcdesktop


train.py --dataset_path ./data/reddit_threads --fp16 O2 --gradient_accumulation_steps 32 --train_batch_size 1 --valid_batch_size 1 --n_epochs 100


- Trying with no authors, with 1-3 candidates (most are 1 naturally so we shall see). And with longer history. It learn to say "no u r"
```
>>> That's a cute kitty                                                                                                                                                                                                                                                        
No u                                                                                                                                                                                                                                                                           
>>> Haha thanks, I'm not cute in real life though                                                                                                                                                                                                                              
You're right, I'm pretty sure that's not the case.                                                                                                                                                                                                                             
>>> Thanks :(
No u
>>> Yeah I get it
Yes, but you're wrong.
>>> No u
Haha
>>> No u
Haha
>>> No u
Haha
>>> No u
No u
>>> No u
No u
>>> No u
No u
>>> No u
No u
>>> No u
No u
>>> No u^[OM
No u
>>> Yes I
No u
>>> Your momma was a truck driver from the moon
Yes, my momma was a truck driver from the moon
>>> But wait
No u
>>> for it
No u
>>> Your father was a bus boy to the stars
No u
>>> And yuour a silly sod
No u
>>> From space
No u
>>> And you suck
No u
>>> haha
No u
>>> q
No u
```

```
Goodbye
>>> RESET
--------------------------------------------------------------------------------
INFO:interact.py:Selected personality: shittyaskscience
#r/lostredditors
>>> Yeah I guess
it's not that bad for me
>>> Why not?
Why not?
>>> I have no arms
Why not?
>>> Yes you do
You are right
>>> Goodbye
Goodbye
>>> RESET
--------------------------------------------------------------------------------
INFO:interact.py:Selected personality: funny
Please don't know, I think he said the same thing.
>>> That's what she said
It’s not. 
>>> Your momma
I've never seen that.
>>> Seen what?
Nothing.
>>> Goodbye
I love it.
>>> RESET
--------------------------------------------------------------------------------
INFO:interact.py:Selected personality: theonion
Yes.
>>> Breaking news and onion was tried last yesterday by the news bear
Yes.
>>> 
Prompt should not be empty!
>>> .
Yes.
>>> Go on
Yes.
>>> Why?
Because we haven't seen it yet
>>> Ok cool, good for you
```

- But it seems to learn small repetitive replies. I also have a lot of data that way, perhaps I should restrict to long replies, karma, or 3 replies
- We could also try the MEDIUM model on colab, and to restrict to one subreddit


TODO:
- [ ] Show how it works each epoch
- [ ] Try GPT-2 medium on colab
