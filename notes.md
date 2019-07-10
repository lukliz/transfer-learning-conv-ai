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
`python -m ipdb train.py --dataset_path ./data/reddit_threads --fp16 O2 --gradient_accumulation_steps 32 --train_batch_size 1 --valid_batch_size 1 --n_epochs 10`

`python interact.py --device cpu --dataset_path ./data/reddit_threads --model_checkpoint runs/Jul09_15-44-47_mjcdesktop/`

python interact.py --dataset_path ./data/reddit_threads --model_checkpoint runs/Jul09_21-25-16_mjcdesktop


train.py --dataset_path ./data/reddit_threads --fp16 O2 --gradient_accumulation_steps 32 --train_batch_size 1 --valid_batch_size 1 --n_epochs 100

`python -m ipdb train.py --dataset_path ./data/reddit_threads --fp16 O2 --gradient_accumulation_steps 32 --train_batch_size 1 --valid_batch_size 1 --n_epochs 10 --min_candidates 3 --num_candidates 3`

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
  - [x] nope can't use with fp16 on 16GB or gpu ram. Pytorch doesn't support TPU's
  - [ ] What about a gcloud K80 with 24GB?

epoch one Jul10_19-23-12_mjcdesktop min and max candidates 3, filtered out moderators and [deleted]
```
Epoch [1/10]: [19447/19447] 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████, loss=1.53e-01 [1:34:29<00:00]
Validation: {'accuracy': 0.22992056487202117,                                                                                                                                               
 'average_accuracy': 0.22992056487202117,
 'average_nll': 3.715492552798981,
 'average_ppl': 41.07881558777738,
 'nll': 3.715492552798981}
INFO:interact.py:Namespace(dataset_cache='./dataset_cache', dataset_path='./data/reddit_threads', device='cpu', max_history=6, max_length=20, min_length=1, model='gpt2', model_checkpoint='runs/Jul10_19-23-12_mjcdesktop', no_sample=False, seed=42, temperature=0.7, top_k=0, top_p=0.9)
INFO:interact.py:Selected personality: /r/rareinsults
human: Hi mother
robot: No problem.
human: Roast me now please just like you roasted your family
robot: Do you have a towel?
human: No, why?
robot: I didn’t know you were a good mother 
human: I'm not, but you wouldn't know anyway, you never had a mother
robot: Yeah, but I’ve always had a father and I’ve always had a son
human: Always had a son, that seem unlikely. Biologically unlikely
robot: Never had a son. 
human: You just said you always did. Are you God?
robot: I am a good mother 
human: God, confimed.
robot: I am a good mother 
human: Yes yes, now get back in the kitchen Norman
robot: I just bought a bag of groceries and it's the same. 
human: RESET
--------------------------------------------------------------------------------
INFO:interact.py:Selected personality: /r/shittyaskscience
robot: ShittyScreamers, they deserve the praise
human: Praise them because thier Shitty Screams keep out screamy shits streaming
robot: I was a little pissed
human: The best way to fix that is to unpiss, best done in the toilet
robot: I was a little pissed
human: Yeah you said that
robot: I was a little pissed
human: RESET
--------------------------------------------------------------------------------
INFO:interact.py:Selected personality: /r/programmingcirclejerk
Nrobot: Because the project is not
human:Now this personality has a lot of data, I hope I get slightly better answers out of you, although it has been only one epoch.
robot: I'm sorry, it's not that I'm not interested in it.
human: What are you interested in brainfuck?
robot: What do you mean?
human: Which language
robot: What do you mean?
human: At least your using proper capitalization aye.       
robot: But what about how you draw a line in the sand?
human: Well my spelling is worse than your so I don't see any line.
robot: And what about your grammatical errors?
human: Yeah they are pretty bad too
robot: I have a lot of it.
human: hmm
robot: You're saying it's easier to write sentences than words?
human: RESET
--------------------------------------------------------------------------------
INFO:interact.py:Selected personality: /r/singularity
robot: I think this is the most accurate
human: I.T. would be more accurate if it was a differen't article?
robot: Yeah, I was trying to write something about the difficulties of predicting the future 
human: Hardly anyone does it properly, Kurzweil actually writes his predictions down, then gets slammed when they are wrong. But everyone who didn't comment to written predictions is more wrong.
robot: We'll just have to wait and see if Kurzweil really knows what he's talking about
human: If he doesn't he will be dead before anyone can reply
robot: What if he's right?
human: Then I will be a suprised little meatsack and you will be dead
robot: You know I have to be careful, my life will be ruined.
--------------------------------------------------------------------------------
INFO:interact.py:Selected personality: /r/rareinsults
human: You cook like you smell, good
robot: What about my tastes?
human: Your face is like your tastes, testically
robot: I know what you mean.
human: I know you do, it's easy. You only need to taste the inside of your mouth, your face, and ...
robot: I know I can't taste the inside of my mouth but I'll get that right here. 
human: Good man!
robot: I'm sorry you're so upset
human: I'm an upset man in and upset world. It's all upside down and turned around. Like your liver.
robot: I was hoping you'd be happy and be able to see the real you.
human: No one knows the real me, not even me
robot: I've never been a stranger to an argument but I'm sure they'll find out

```
