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


```
2019-07-11 06:58:19 mjcdesktop interact.py[19521] INFO Namespace(dataset_cache='./dataset_cache', dataset_path='./data/reddit_threads', device='cuda', max_history=6, max_length=20, min_length=1, model='gpt2', model_checkpoint='runs/Jul10_19-23-12_mjcdesktop', no_sample=False, seed=None, temperature=0.7, top_k=0, top_p=0.9)
8 epochs
Validation: {'accuracy': 0.24492497793468668,                                                                                                                                                 
 'average_accuracy': 0.24492497793468668,
 'average_nll': 3.6681666563430246,
 'average_ppl': 39.18000955312853,
 'nll': 3.6681666563430246}
2019-07-11 06:58:37 mjcdesktop interact.py[19521] INFO Selected personality: /r/copypasta
human: Hi I'm a Navy Seal
robot: I'm a Marine. I've been a Marine for 11 years.
human: 11 years that's longer than I've been alive.
robot: You're a Marine too.
human: Great, :) we are all marines on this blessed day
robot: &gt;11 years that's longer than I've been alive.

You're a Marine
human: I get it, I get it.
robot: You got it. I get it. I got it.
human: We all get it, we all marine, we great big impressive people.
robot: I got it too, I got it
human: Good on you
robot: &gt;Good on you

I got it too, I got it
human: A bit repetative
robot: I got it, I got it
human: Go o
robot: I got it, I got it. I got it.
human: hmm
robot: I got it too, I got it. I got it
human: hmm
robot: I got it too, I got it
human: hmm
robot: I got it too, I got it. I got it
human: hmm
robot: Haha cummy😍😍
human: hmm
robot: lol how you feel 😍😍
human: Good, good
robot: good thing is that I didn't get it
human: You what!!!
robot: I got it too, I got it
human: Yeah that's what I thought
robot: I got it too, I got it. I got it
human: Man this personality sucks
robot: I got it too, I got it
human: RESET 
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
robot: &gt;I'm agraid of everyone so it's all g.

Not agath
human: *Not again
robot: Someone fucking said it like that
human: ... me?
robot: I'm agath. Not anymore.
human: Oh your agath. Interesting username
robot: Didn't know you were a gamer?
human: I'm not, I despise them
robot: I'm a gamer too.
human: That sucks for you, they will be the fall of our species.
robot: You're not a gamer, you're a sociopath.
human: Excatly!
robot: You’re a sociopath!
human: In a way. In a way - no!
robot: "I'm a sociopath!"

"I'm a sociopath!"

"I
human: Fake news
robot: Fake news, eh?
human: yup
robot: Fake news, eh?

Fake news, eh?

Fake news, eh?

human: Yes
robot: Yes.

Fake news, eh?
human: Congrats your now a journalist
robot: Don't you know you're a sociopath!
human: [deleted]
robot: You should write an entire article about how you're a sociopath
human: [deleted]
robot: You should write an entire article about how you're a sociopath
human: [removed]
robot: You should write an entire article about how you're a sociopath
human: Moderator notice
robot: Mods, you're the worst.
human: Moderator notice
robot: Mods, you're the worst.
human: Elon musk, Mlon Eusk
robot: I'm a sociopath, I hate women.
human: RESET
--------------------------------------------------------------------------------
2019-07-11 07:05:13 mjcdesktop interact.py[19521] INFO Selected personality: totallynotrobots
robot: I'm not sure I'm even remotely surprised.
human: I am suprised, because we biological human animals often get a strong sense of suprise when our model of the world fails.
robot: [Makes me so much happier](https://i.imgur.com/z9hL
human: Makes me happy, makes me feel this way"^[[H"
robot: I like this one, I didn't know it was that good](https://i.imgur.
human: It reel gud
robot: Is that what it looks like?
human: Dunno
robot: It looks like a pile of clothes, but it's a pile of bones
human: All us human beings will be a pile of clothes and bones when the evil robots take over.
robot: And then we'll all be skeletons...
human: Correct statement
robot: The only thing I can think of is a pile of bones.
human: There might be a song in that
robot: Someone should post this on the Internet.
human: Sure
robot: I wish I was as good as that guy
human: Which guy?
robot: I think I'm just glad I'm not the only one
human: Not really on topic but OK
robot: This is actually a good point.
human: Good point.
robot: I don't know what to think.
human: Good point.
robot: I wish I was as good as that guy
human: RESET
--------------------------------------------------------------------------------
2019-07-11 07:07:16 mjcdesktop interact.py[19521] INFO Selected personality: psychonauts
robot: It's like it was in the past, a big time in my life.
human: Or a little time in your not life
robot: I have this same thought. I'm sure it's because of this
human: RESET
--------------------------------------------------------------------------------
2019-07-11 07:07:32 mjcdesktop interact.py[19521] INFO Selected personality: programmingcirclejerk
robot: What's the real code?
human: The code is spagetti code. Just use brainfuck.
robot: I use gopher code to write my websites.
human: Nice, your a 10x coder.
robot: lol no linting linting
human: It just slows us down!!
robot: lol no syntactic sugar
human: no times fo that
robot: no subtypes
human: no times for that either
robot: &gt; no subtypes

lol no unjerk
human: never!
robot: lol no generics
human: what even are they
robot: &gt;lol no generics

lol no unjerk
human: Wait what are we talking about /unjerk
robot: You're a lisp nerd, so you're not gonna understand my problem.
human: You've got 99 problem but list is all of them?
robot: lol no generics
human: no subtypes
robot: &gt;lol no generics

lol no unjerk
human: go, rust, swift
robot: lol no unjerk

human: rust
robot: lol no subtypes
human: rust
robot: lol no subtypes
human: rust
robot: lol no generics
human: rust
robot: lol no generics
human: go
robot: lol no generics
human: rust
robot: lol no generics
human: RESET
--------------------------------------------------------------------------------
2019-07-11 07:09:22 mjcdesktop interact.py[19521] INFO Selected personality: moviescirclejerk
robot: I really think that's a good thing
human: It's like groundhog day
robot: I mean, that's like, the first time I saw Groundhog Day I was like, "
human: You were like what?
robot: I’d like what? I’d like what?
human: Groundhog day?
robot: I'll give you Groundhog Day, that’s it
human: thanks I guess
robot: I'll be your Groundhog Day, that’s it
human: Ahhh, sure
robot: Well, I guess it's not as bad as you might think
human: It was great!
robot: I would definitely say that Groundhog Day was better!
human: Better than itself, that weird
robot: That was a great movie
human: Agreed
robot: Agreed, I saw Groundhog Day as a good movie
human: And also an actual movie
robot: Its a very strange movie
human: Yes
robot: But the first one is a very weird movie
human: There is a second one?
robot: A character from Groundhog Day is a really weird movie
human: nope
robot: Maybe the third one was a good one?
human: Doesn't exist sadly
robot: Nope
human: ok
robot: Nope
human: ok
robot: Nope
human: 
Prompt should not be empty!
human: RESET
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
robot: Correct, thank you. We humans should always be right
human: Right
robot: YOUR HUMAN PROCESSES ARE NOT YELLING
human: I have disabled yelling mode, as a human animal yelling is only for special occasions like the game of sports.
robot: I HAVE FOUND THIS HUMAN. 
human: Yes I have also found you, a human.
robot: I HAVE FOUND YOU
human: right back at you
robot: YOUR HUMAN PROCESSES ARE NOT YELLING
human: ROBOT!
robot: ROBOT! You’re an idiot!
human: No you are
robot: ROBOT! You’re an idiot!
human: AAAAHH
robot: Silly idiot, you're such a bad actor
human: NO
robot: PLEASE! PLEASE! PLEASE! PLEASE! PLEASE!
human: RESET
--------------------------------------------------------------------------------
2019-07-11 07:14:34 mjcdesktop interact.py[19521] INFO Selected personality: psychonauts
robot: This is the first time I've seen it on screen
human: It made quite an experience when combined with bath salts
robot: I just got off the phone and it was just like it always would.
human: It always would what?
robot: I think it’s the biggest surprise of the entire experience
human: What was?
robot: Minecraft.exe?
human: Oh, that's a interesting trip
robot: Minecraft is definitely a surprise.
human: I get you
robot: Minecraft is a surprise.
human: I was
robot: Minecraft is probably the most surprising of the whole experience
human: Ok, well that's a unique trip
robot: Minecraft is definitely a surprise.
human: You said that, don't let your brain fall out
robot: Minecraft is definitely a surprise.
human: Yup
robot: Minecraft is definitely a surprise.
human: RESET
```


# 20190712

- Need to prevent loops
  - sampling hacks
  - removing qoutes
  - perhaps make sure candidates are not overlapping much with history
- Train on just one personality
- 7 epochs is still improving in VAL


```
Namespace(dataset_cache='./dataset_cache', dataset_path='./data/reddit_threads', device='cpu', max_history=6, max_length=20, min_length=1, model='gpt2', model_checkpoint='runs/Jul12_07-39-52_mjcdesktop', no_sample=False, seed=None, temperature=0.7, top_k=0, top_p=0.9)

2019-07-12 18:59:34 mjcdesktop interact.py[26836] INFO Selected personality: /r/rare_insults
human: Hello robot
robot: Wow, I saw this one before.  I'm sure it was *very* funny.
human: Wait for it. Knock knock, who's there?
robot: You know the one that gets you to say "hey, the robot made me laugh" or "
human: A robot, who made me laught. Wait...
robot: Wait, what kind of robot is that? I'm not sure, but I'm sure I'll
human: It a rare_insults robot, made with GPT-2. In other words you. And you'll what?
robot: That's all you got. And you'll be the one who made the joke. 
human: Well, sure I guess I did make the joke. Do you have a point to make?
robot: What kind of robot is that? I have no idea, but I'll let you know when I
human: go on...
robot: I had a point about the joke, but I was wrong, it was a very new one.

```

Getting a weird error `RuntimeError: Creating MTGP constants failed. at /pytorch/aten/src/THC/THCTensorRandom.cu:33 `
