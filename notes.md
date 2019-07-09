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
<!-- - [ ] refactor and load in term -->
- [ ] change formatting
- [ ] remove the ids, we don't need it in this format
