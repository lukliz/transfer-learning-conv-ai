import collections
import copy
import itertools
import pickle
from pathlib import Path
import logging

from anytree import Node
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm

logger = logging.getLogger(__file__)

def format_thing(thing, submission_id):
    if thing["type"] == "submission":
        return (
            "****S\n"
            + "\n".join([thing["url"], thing["title"], thing.get("selftext", "")])
            + "\n****ES "
            + thing["id"]
            + "\n"
        )
    elif thing["parent_id"] == submission_id:
        return (
            "****T "
            + thing["parent_id"][3:]
            + "\n"
            + thing["body"]
            + "\n****ET "
            + thing["id"]
            + "\n"
        )
    else:
        return (
            "****R "
            + thing["parent_id"][3:]
            + "\n"
            + thing["body"]
            + "\n****ER "
            + thing["id"]
            + "\n"
        )


def get_id_for_comments(thing):
    if thing["type"] == "submission":
        return "t3_" + thing["id"]
    else:
        return "t1_" + thing["id"]


def thread2tree(comment_dict, submission):
    """Convert list of reddit comments and a submission to a tree."""
    comment_dict = copy.deepcopy(comment_dict)
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

    # tree from comments dict
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


def get_dataset(tokenizer, data_path):
    """
    Load pickle files with reddit comments.

    Structure:
    - valid list
    - dict
        - personality: list[str]
        - utterances: list
            - dict:
                - candidates: list[str]
                - history: list[str]
    
    """
    data_dir = Path(data_path)
    subreddit_paths = [d for d in data_dir.glob("*/") if d.is_dir()]

    # collect data by subreddit
    splits = dict(train={}, valid={}, test={})
    for subreddit_path in subreddit_paths:
        subreddit_files = sorted(subreddit_path.glob("*.pickle"))
        subreddit = subreddit_path.name

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

    # collect data into the same dict format as hugging face
    dataset2 = collections.defaultdict(list)
    for split, personalities in splits.items():
        total = len(list(itertools.chain(*personalities.values())))
        with tqdm(total=total, desc=f'{split}', unit='file') as prog:
            for personality, files in personalities.items():
                utterances = []
                for file in files:
                    # load
                    thread = pickle.load(file.open("rb"))
                    prog.update(1)
                    try:
                        nodes_by_id, thing_by_id = thread2tree(thread["comment_dict"], thread["submission"])
                    except Exception as e:
                        logger.warn("Exception for file %s, %s", file, e)
                        continue
                    
                    # get utterances
                    min_candidates = 3
                    submission_id = get_id_for_comments(thread['submission'])
                    for current_node in nodes_by_id.values():
                        if (
                            current_node.parent
                            and len(current_node.path) > 1
                            and len(current_node.children) >= min_candidates
                        ):
                            history = [
                                format_thing(thing_by_id[node.name], submission_id)
                                for node in current_node.path
                            ]

                            candidates = [
                                format_thing(thing_by_id[node.name], submission_id)
                                for node in current_node.children
                            ]
                            # FIXME (wassname), this repo seems to be factored for a static number of candidates per personality so lets clip at 3
                            candidates = candidates[:min_candidates]

                            utterance = dict(candidates=candidates, history=history)
                            utterances.append(utterance)
                        else:
                            logger.debug("skipping node with too few paths")
                dataset2[split].append(dict(personality=[personality], utterances=utterances))

    logger.info("Tokenize and encode the dataset")

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(obj)[:tokenizer.max_len]
            )
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    dataset2 = tokenize(dataset2)
    return dataset2
