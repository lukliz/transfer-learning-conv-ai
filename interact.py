# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
from pprint import pformat

import coloredlogs
import crayons
import torch
import torch.nn.functional as F

from data import MJC_FINETUNED_MODEL, download_targz_to_folder
from pytorch_pretrained_bert import (GPT2LMHeadModel, GPT2Tokenizer,
                                     OpenAIGPTLMHeadModel, OpenAIGPTTokenizer)
from train import SPECIAL_TOKENS, build_input_from_segments

coloredlogs.install()


def top_filtering(
    logits, top_k=0, top_p=0.0, threshold=-float("Inf"), filter_value=-float("Inf")
):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert (
        logits.dim() == 1
    )  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(
            personality, history, current_output, tokenizer, with_eos=False
        )

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(
            instance["token_type_ids"], device=args.device
        ).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = (
            torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        )
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def run():
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model type (gpt or gpt2)"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="",
        help="Path, url or short name of the model",
    )
    parser.add_argument(
        "--max_history",
        type=int,
        default=4,
        help="Number of previous utterances to keep in history",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Set to use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum length of the output utterances",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=20,
        help="Minimum length of the output utterances",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed")
    parser.add_argument(
        "--temperature", type=int, default=0.7, help="Sampling softmax temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Filter top-k tokens before sampling (<=0: no filtering)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.6,
        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        args.model_checkpoint = download_targz_to_folder(MJC_FINETUNED_MODEL)

    if args.seed is not None:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()

    logger.info("Sample a personality")
    model_training_args = Path(args.model_checkpoint).joinpath(
        "model_training_args.bin"
    )
    training_args = torch.load(model_training_args.open("rb"))
    personalities_str = getattr(training_args, "subreddit", [])
    personalities = [
        [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))]
        for obj in personalities_str
    ]
    if not personalities:
        raise FileNotFoundError(
            f"Could not load personalities from file {model_training_args}"
        )
    personality = random.choice(personalities)
    print("personalities", [tokenizer.decode(chain(*p)) for p in personalities])
    logger.info("Selected personality: /r/%s", tokenizer.decode(chain(*personality)))

    history = []
    while True:
        raw_text = input(f"{crayons.green('>>> ')}")
        while not raw_text:
            print(f"\n{crayons.red('Prompt should not be empty!')}")
            raw_text = input(f"{crayons.green('>>> ')}")
        history.append(tokenizer.encode(raw_text))

        if raw_text == "RESET":
            print("-" * 80)
            history = []
            personality = random.choice(personalities)
            logger.info(
                "Selected personality: /r/%s", tokenizer.decode(chain(*personality))
            )

        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2 * args.max_history + 1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(f'{crayons.blue("robot:")}{out_text}')


if __name__ == "__main__":
    run()
