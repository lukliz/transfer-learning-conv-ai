import collections
import copy
import html
import itertools
import pickle
import logging
import random
import tarfile
import tempfile
from fuzzywuzzy import fuzz
from pathlib import Path

import simple_cache
from anytree import Node
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm

from pytorch_pretrained_bert import cached_path

logger = logging.getLogger(__file__)


PERSONACHAT_URL = "http://publicmldatasets.thinkcds.com/transfer-learning-conv-ai/20190715_reddit_threads_pickle.tar.gz"
MJC_FINETUNED_MODEL = "http://publicmldatasets.thinkcds.com/transfer-learning-conv-ai/Jul13_18-24-35_mjcdesktop.tar.gz"

logger = logging.getLogger(__file__)


def download_targz_to_folder(url):
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(url)
    tempdir = tempfile.mkdtemp()

    logger.info(
        "extracting archive file {} to temp dir {}".format(
            resolved_archive_file, tempdir
        )
    )
    with tarfile.open(resolved_archive_file, "r:gz") as archive:
        archive.extractall(tempdir)
    return tempdir


def format_reddit_thing(thing, submission_id):
    """Format a dict of comment or submisson data."""

    if thing["type"] == "submission":
        text = "\n".join([thing["title"], thing.get("selftext", "")])
    else:
        text = thing["body"]
    text = html.unescape(text)
    return text


def get_id_for_comments(thing):
    if thing["type"] == "submission":
        return "t3_" + thing["id"]
    else:
        return "t1_" + thing["id"]


def thread2tree(comment_dict, submission):
    """
    Convert list of reddit comments and a submission to a tree.

    Comment dict is {'t1_frg5g': {"body":"This is a comment", ....}}
    Submission is {'id':'t3_th5gf', 'selftext': 'This is the OP', ...}
    """
    comment_dict = copy.deepcopy(comment_dict)

    # Sort comments by their parent id
    queue = [submission]
    while len(list(itertools.chain(*comment_dict.values()))) > 0:
        for queue_position in range(len(queue) - 1, -1, -1):
            current_id = get_id_for_comments(queue[queue_position])
            found = comment_dict[current_id]
            if len(found):
                break
        next_comment = comment_dict[current_id].pop()
        queue.append(next_comment)

    # convert thread to persona type file. Where persona is submision
    submission_id = get_id_for_comments(submission)

    # Make a tree from comments dict
    tree_root = Node(submission_id)
    submission_id = get_id_for_comments(submission)
    nodes_by_id = {submission_id: tree_root}
    thing_by_id = {submission_id: submission}
    for thing in queue[1:]:
        parent = nodes_by_id[thing["parent_id"]]
        n = Node(get_id_for_comments(thing), parent=parent)
        nodes_by_id[get_id_for_comments(thing)] = n
        thing_by_id[get_id_for_comments(thing)] = thing
    return nodes_by_id, thing_by_id


def collect_thread_files(data_dir, subreddits):
    """Collect pickle thread files and split into train, val, test.
    
    """
    subreddit_paths = [d for d in data_dir.glob("*/") if d.is_dir()]

    # collect data by subreddit
    splits = dict(train={}, valid={}, test={})
    for subreddit_path in subreddit_paths:
        subreddit_files = sorted(subreddit_path.glob("*.pickle"))
        if len(subreddit_files) > 10:
            subreddit = subreddit_path.name
            if (subreddits == []) or (subreddit in subreddits):
                print(f"{len(subreddit_files):10d} threads from /r/{subreddit}")

                # split
                train_files, test_files = train_test_split(
                    subreddit_files, test_size=0.1, random_state=42
                )
                train_files, valid_files = train_test_split(
                    train_files, test_size=0.1, random_state=42
                )

                splits["train"][subreddit] = train_files
                splits["valid"][subreddit] = valid_files
                splits["test"][subreddit] = test_files
            else:
                print(f"subreddit not in filter /r/{subreddit}")

    num_train_examples = len(list(itertools.chain(*list(splits["train"].values()))))
    if len(splits["train"]) == 0 or num_train_examples < 10:
        raise Exception(
            f"not enougth training data found in '{data_dir}'. Check your dataset_path and your --subreddits argument"
        )
    return splits


def cache_load_utturances(ttl=360000):
    """
    Decorator for wrapping simple cache around load_utterances.

    Since some arguments are unhashable (tokenizer) or immutable (list) we need to make the key manually
    """

    def decorate(func):
        @simple_cache.wraps(func)
        def wrapper(**kwargs):
            # key = (args, tuple_kwargs(kwargs))
            filename = f"data/.simple.{kwargs['personality']}.cache"
            tokenizer = kwargs["tokenizer"]
            # We must use immutaable, hashable args as keys, so no lists, sets, or tokenizer
            key = simple_cache.tuple_kwargs(
                dict(
                    personality=kwargs["personality"],
                    max_seq_len=kwargs["max_seq_len"],
                    files=tuple(sorted([str(f) for f in kwargs["files"]])),
                    tokenizer_name=type(tokenizer).__name__,
                    vocab_size=len(tokenizer.encoder),
                    special_tokens=tuple(sorted(tokenizer.special_tokens)),
                    num_candidates=kwargs["num_candidates"],
                )
            )
            value = simple_cache.load_key(filename, key)
            if value is None:
                value = func(**kwargs)
                simple_cache.save_key(filename, key, value, ttl)
            else:
                print(f'Loaded utturances from cache for {kwargs["personality"]}')
            return value

        return wrapper

    return decorate


def authors2ints(authors):
    # e.g. authors = ['paul', 'dan', 'mike', 'iv', 'iv', 'dan']
    author2int = dict((v,k) for k,v in enumerate(set(authors)))
    return [str(author2int[author]) for author in authors]

@cache_load_utturances()
def load_utterances(personality, files, tokenizer, max_seq_len, num_candidates=10):
    utterances = []
    for file in tqdm(files, desc=f"Loading {personality}", unit="thread"):
        # load
        try:
            thread = pickle.load(file.open("rb"))
        except Exception as e:
            logger.warning(f"Exception opening {file}, {e}")
            continue

        # Anytree seems to be v. slow of theads with lots of comments (>1000)
        comments_all = len(
            list(itertools.chain(*list(thread["comment_dict"].values())))
        )
        if comments_all > 1000:
            logger.debug(
                f"Skipping {personality} thread with many ({comments_all}) comments"
            )
            continue
        try:
            nodes_by_id, thing_by_id = thread2tree(
                thread["comment_dict"], thread["submission"]
            )
        except IndexError as e:
            logger.warn("IndexError for file '%s', '%s'", file, e)
            continue
        except KeyError as e:
            logger.warn("KeyError for file '%s', '%s'", file, e)
            continue

        # get utterances
        # Make a max number of candidates that will fit on your GPU
        submission_id = get_id_for_comments(thread["submission"])
        for current_node in nodes_by_id.values():
            if (
                current_node.parent
                and len(current_node.path) > 1  # It must have some parent comments
                and len(current_node.children) >= 1  # And child comments
            ):
                history_things = [thing_by_id[node.name] for node in current_node.path]
                history = [
                    format_reddit_thing(thing, submission_id)
                    for thing in history_things
                ]

                # Remember each author, but as an int
                authors = authors2ints([thing['author'] for thing in history_things])

                replies = [thing_by_id[node.name] for node in current_node.children]

                # We now want to find distractors. None of these ID's will do
                correct_ids = (
                    [node.name for node in current_node.path]
                    + [submission_id]
                    + [node.name for node in current_node.children]
                )
                distractor_ids = [
                    k for k, v in nodes_by_id.items() if k not in correct_ids
                ]
                distractors = [thing_by_id[d_id] for d_id in distractor_ids]

                # Filter some of the bad data out, yeah it's a hack, but some is very usefull
                filters = [
                    lambda x: "[deleted]" not in x.get("body", ""),
                    lambda x: "[removed]" not in x.get("body", ""),
                    lambda x: x.get("author", "") != "[removed]",

                    # Filter out the repetative mod and sticky comments
                    lambda x: x.get("author", "") != "AutoModerator",
                    lambda x: not x.get("stickied", False),

                    # Short comments are low information and too easy
                    lambda x: len(x.get("body", "")) > 30,
                    lambda x: len(x.get("body", ""))
                    < 240,  # Ones that are too long don't do well sometimes, lets keep it tweet length

                    # Also filter out op? In roast me they do not do roasting
                    lambda r: r['author']!=history_things[0]['author'],

                    # the output tends to be repetitive and loop, lets avoid that a bit by filtering out v. repetitive replies
                    lambda x: max([fuzz.ratio(x.get("body", ""), h)/100 for h in history]) < 0.75
                ]
                for f in filters:
                    replies = filter(f, replies)
                    distractors = filter(f, distractors)

                # Format "things" from reddit
                replies = [
                    format_reddit_thing(thing, submission_id) for thing in replies
                ]
                distractors = [
                    format_reddit_thing(thing, submission_id) for thing in distractors
                ]

                # # also removed qouted text
                # replies = [re.sub('&gt;.*\n', '', r) for r in replies]
                # distractors = [re.sub('&gt;.*\n', '', r) for r in distractors]

                if len(distractors) >= num_candidates - 1:
                    # Distractors at start of candidates, real reply at end
                    for reply in replies:
                        candidates = random.sample(distractors, num_candidates - 1) + [
                            reply
                        ]

                        utterance = dict(candidates=candidates, history=history, authors=authors)
                        utterance = tokenize(utterance, tokenizer, max_seq_len)
                        utterances.append(utterance)
            else:
                logger.debug("skipping node with too few paths")

    personality_toks = tokenize([personality], tokenizer, max_seq_len)
    return dict(personality=personality_toks, utterances=utterances)


def threads_to_utterances(splits, tokenizer, max_seq_len):
    """Process a json of personality threads into utterances.
    
    json structure:
    - valid list
    - dict
        - personality: list[str]
        - utterances: list
            - dict:
                - candidates: list[str]
                - history: list[str]
    """
    # collect data into the same dict format as hugging face
    dataset2 = collections.defaultdict(list)
    for split, personalities in splits.items():
        for personality, files in personalities.items():
            utterances_dict = load_utterances(
                personality=personality,
                files=files,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                num_candidates=10,
            )

            dataset2[split].append(utterances_dict)

            logger.info(
                f"Utterances for {split} & /r/{personality}: {len(utterances_dict['utterances'])}"
            )
    return dataset2


def get_dataset(tokenizer, data_path, subreddits=[], max_seq_len=None):

    max_seq_len = max_seq_len or tokenizer.max_len
    if data_path == "":
        data_path = download_targz_to_folder(PERSONACHAT_URL) + "/reddit_threads"
    data_dir = Path(data_path)
    print('data_dir', data_dir)

    splits = collect_thread_files(data_dir, subreddits)

    dataset2 = threads_to_utterances(splits, tokenizer, max_seq_len)
    return dataset2


def tokenize(obj, tokenizer, max_seq_len):
    """Recursively convert to tokens."""
    if isinstance(obj, str):
        toks = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj)[:max_seq_len])
        assert all(
            [t < len(tokenizer.encoder) for t in toks]
        )  # all(toks < len(tokenizer.encoder))
        return toks
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer, max_seq_len)) for n, o in obj.items())
    return list(tokenize(o, tokenizer, max_seq_len) for o in obj)
