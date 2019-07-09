- get data with `1_fetch_pushshift_data` (run overnight)
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
    - [ ]  fix the seq length, it seems that I need to concat when adding hist etc
- [ ] change formatting
  - [ ] remove the ids, we don't need it in this format
- [ ] interact
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
