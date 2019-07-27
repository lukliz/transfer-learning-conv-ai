# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import logging
import math
import gc
import os
import sys
import itertools
import datetime
import functools
import random
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from pprint import pformat

import coloredlogs
import torch
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import (
    OptimizerParamsHandler,
    OutputHandler,
    TensorboardLogger,
)
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage, EpochMetric
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from data import get_dataset
from pytorch_pretrained_bert import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    GPT2DoubleHeadsModel,
    GPT2Tokenizer,
    OpenAIAdam,
    OpenAIGPTDoubleHeadsModel,
    OpenAIGPTTokenizer,
)

SPECIAL_TOKENS = ["<bos>", "<eos>", "<spartner>", "<sother>", "<sself>", "<pad>"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)


def clear_mem():
    # Clear cache to avoid memory overflow on eval step
    gc.collect()
    torch.cuda.empty_cache()


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = (
        torch.tensor(scalar, dtype=torch.float, device=args.device)
        / torch.distributed.get_world_size()
    )
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [
            x + [padding if name != "lm_labels" else -1] * (max_l - len(x))
            for x in dataset[name]
        ]
    return dataset


def _truncate_seq_pair_n(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    # from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier_dataset_utils.py#L482
    while True:
        total_length = sum(len(s) for s in tokens)
        if total_length <= max_length:
            break
        longest = sorted(tokens, key=len, reverse=True)[0]
        longest.pop()


def build_input_from_segments(
    persona,
    history,
    reply,
    authors,
    tokenizer,
    lm_labels=False,
    with_eos=True,
    max_len=None,
):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker_partner, speaker_other, speaker_self = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1]
    )
    if max_len is None:
        max_len = tokenizer.max_len

    # First make sure history add up to max seq len, we do this seperatly so that history is cropped before reply
    _truncate_seq_pair_n(history, int(max_len / 3 * 2))
    instance = {}
    sequence = [list(chain(*persona))] + history + [reply]

    # Clip to by removing from longest message. Clip to max len len minus special tokens that will be added.
    _truncate_seq_pair_n(sequence, max_len - len(history) - 2 - with_eos)
    persona_c = sequence[0]
    history_c = sequence[1:-1]
    reply_c = sequence[-1]

    # Convert authors to tokens
    author2token = {}
    if len(authors) > 0:
        author2token[authors[-1][0]] = speaker_partner
    if len(authors) > 1:
        author2token[authors[-2][0]] = speaker_self
    author_tokens = [author2token.get(author[0], speaker_other) for author in authors]

    # Add author tokens
    history_c = [[author_tokens[i]] + history_c[i] for i in range(len(history_c))]
    reply_c = [speaker_self] + reply_c

    # add it all together into full seq
    sequence = [[bos] + persona_c] + history_c + [reply_c + ([eos] if with_eos else [])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [
        sequence[i][0] for i, s in enumerate(sequence) for _ in s
    ]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = (
            ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        )

    if len(instance["input_ids"]) > max_len:
        logger.warn(
            f'input should be less than max len {len(instance["input_ids"])} < {max_len}'
        )
    assert (
        len(instance["input_ids"]) <= max_len
    ), f'input should be less than max len {len(instance["input_ids"])} < {max_len}'
    return instance, sequence


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(
        tokenizer,
        args.dataset_path,
        subreddits=args.subreddit,
        max_seq_len=args.max_seq_len,
        mimic_op=args.mimic_op,
    )
    if not personachat:
        raise ValueError("No dataset loaded")

    logger.info("Build inputs and labels")
    datasets = {
        "train": defaultdict(list),
        "valid": defaultdict(list),
        "test": defaultdict(list),
    }
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0:  # and dataset_name == "train":
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2 * args.max_history + 1) :]
                authors = utterance["authors"][-(2 * args.max_history + 1) :]
                for j, candidate in enumerate(
                    utterance["candidates"][-num_candidates:]
                ):
                    # The last candiate is the real answer, the others are distactors
                    #  used for the classification part of the dual language model
                    lm_labels = bool(j == num_candidates - 1)
                    instance, _ = build_input_from_segments(
                        persona,
                        history,
                        candidate,
                        authors,
                        tokenizer,
                        lm_labels,
                        max_len=args.max_seq_len,
                    )
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                datasets[dataset_name]["n_candidates"] = num_candidates

    # Preview inputs
    for key in datasets[dataset_name].keys():
        value = datasets[dataset_name][key]
        if isinstance(value, list):
            value = value[-2:]
        logger.debug(f"{key} {value}")

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": [], "test": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(
            dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
        )
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view(
                    (-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:]
                )
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = (
        TensorDataset(*tensor_datasets["train"]),
        TensorDataset(*tensor_datasets["valid"]),
    )

    if args.distributed:
        train_dataset = train_dataset[: args.max_epoch_length]

        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset)
            if args.distributed
            else None
        )
        valid_sampler = (
            torch.utils.data.distributed.DistributedSampler(valid_dataset)
            if args.distributed
            else None
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=args.max_epoch_length
        )
        valid_sampler = torch.utils.data.RandomSampler(
            valid_dataset, replacement=True, num_samples=int(args.max_epoch_length // 8)
        )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        # shuffle=(not args.distributed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=args.valid_batch_size,
        shuffle=False,
    )

    logger.info(
        "Train dataset (Batch, Candidates, Seq length): {}. Batches: {}".format(
            train_dataset.tensors[0].shape, len(train_dataset)
        )
    )
    logger.info(
        "Valid dataset (Batch, Candidates, Seq length): {}, Batches: {}".format(
            valid_dataset.tensors[0].shape, len(valid_dataset)
        )
    )
    assert (
        train_dataset.tensors[0].shape[2] <= args.max_seq_len
    ), "sequences should be less than max len"
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path or url of the dataset. If empty download from S3.",
    )
    parser.add_argument(
        "-s",
        "--subreddit",
        type=str,
        action="append",
        default=[],
        help="Limit the subreddits you train on",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="gpt2",
        help="Path, url or short name of the model",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=2,
        help="Number of candidates for training. Larger numbers may not fit on your GPU",
    )
    parser.add_argument(
        "--max_history",
        type=int,
        default=4,
        help="Number of previous exchanges to keep in history",
    )
    parser.add_argument(
        "--max_epoch_length", type=int, default=100000000000, help="Limit epoch length"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=1, help="Batch size for validation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Accumulate gradients on several steps",
    )
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument(
        "--lm_coef", type=float, default=1.0, help="LM loss coefficient"
    )
    parser.add_argument(
        "--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient"
    )
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Clipping gradient norm"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_before_start",
        action="store_true",
        help="If true start with a first evaluation before training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--fp16",
        type=str,
        default="",
        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation). Try O2. Note first char is the letter 'oh'",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (-1: not distributed)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Max length size, same or smaller than n_ctx in model",
    )
    parser.add_argument(
        "--mimic_op",
        type=bool,
        default=None,
        help="Whether training should train only on replies where the original poster is author (in contrast False means only on non OP replies). Default none will do all replies",
    )

    args = parser.parse_args()

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H-%M-%S")
    model_type_name = "gpt2" if "gpt2" in args.model_checkpoint else "gpt"
    logdir = Path(f"runs/{ts}_{model_type_name}")
    logdir.mkdir()

    logging.basicConfig(
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        # format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=f"{logdir}/train_{args.local_rank}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    coloredlogs.install(
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning(
        "Running process %d", args.local_rank
    )  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = args.local_rank != -1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    logger.info("Prepare tokenizer - add special tokens for fine-tuning")
    tokenizer_class = (
        GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer
    )
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.set_special_tokens(SPECIAL_TOKENS)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(
        args, tokenizer
    )

    logger.info(
        "Prepare pretrained model and optimizer - add special tokens for fine-tuning"
    )
    model_class = (
        GPT2DoubleHeadsModel
        if "gpt2" in args.model_checkpoint
        else OpenAIGPTDoubleHeadsModel
    )
    model = model_class.from_pretrained(args.model_checkpoint)
    model.set_num_special_tokens(len(SPECIAL_TOKENS))
    model.to(args.device)
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.n_epochs
    optimizer = OpenAIAdam(model.parameters(), lr=args.lr)
    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        lm_loss, mc_loss = model(*batch)
        loss = (
            (lm_loss * args.lm_coef + mc_loss * args.mc_coef)
            / args.gradient_accumulation_steps
            / args.train_batch_size
        )
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch, log_output=False):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            model_outputs = model(
                input_ids, mc_token_ids, token_type_ids=token_type_ids
            )
            lm_logits, mc_logits = (
                model_outputs[0],
                model_outputs[1],
            )  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = (
                lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            )
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            # Every now and again sample I guess I should make a custom engine for this
            if log_output:
                input_text = tokenizer.decode(
                    input_ids[0, -1, :].cpu().tolist()
                ).rstrip("<pad>")
                output_text = tokenizer.decode(
                    lm_logits[0, -1, :].argmax(-1).cpu().tolist()
                ).strip()[:200]
                logger.info("inputs : %s", input_text)
                logger.info("outputs: %s", output_text)
            return dict(
                lm_logits_flat_shifted=lm_logits_flat_shifted,
                mc_logits=mc_logits,
                lm_labels_flat_shifted=lm_labels_flat_shifted,
                mc_labels=mc_labels,
                lr=torch.Tensor([optimizer.get_lr()[0]]),
            )

    evaluator = Engine(inference)
    exampler = Engine(functools.partial(inference, log_output=True))

    trainer.add_event_handler(Events.EPOCH_STARTED, lambda _: clear_mem())
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda _: clear_mem())

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader)
    )
    # After eval, run a short engine that will log some examplts
    evaluator.add_event_handler(
        # Events.EPOCH_COMPLETED, lambda _: exampler.run([next(iter(val_loader))])
        Events.EPOCH_COMPLETED,
        lambda _: exampler.run(itertools.islice(val_loader, 2)),
    )
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: train_sampler.set_epoch(engine.state.epoch),
        )
        evaluator.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: valid_sampler.set_epoch(engine.state.epoch),
        )

    # Learning rate warms up then linearly decreases
    tot_iters = args.n_epochs * len(train_loader)
    scheduler = PiecewiseLinear(
        optimizer, "lr", [(0, 0), (int(tot_iters * 0.3), args.lr), (tot_iters, 0.0)]
    )
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {
        "nll": Loss(
            torch.nn.CrossEntropyLoss(ignore_index=-1),
            output_transform=lambda x: (
                x["lm_logits_flat_shifted"],
                x["lm_labels_flat_shifted"],
            ),
        ),
        # Display the lr for each epoch, using the metrics api
        "lr": EpochMetric(
            output_transform=lambda x: (x["lr"], x["lr"]),
            compute_fn=lambda x, y: x[0].mean(),
        ),
    }
    # Meta metrics
    metrics.update(
        {"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)}
    )
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])

    # Only add accuracy if are using distractors
    if args.num_candidates > 1:
        metrics["accuracy"] = Accuracy(
            output_transform=lambda x: (x["mc_logits"], x["mc_labels"])
        )
        metrics.update(
            {
                "average_accuracy": MetricsLambda(
                    average_distributed_scalar, metrics["accuracy"], args
                )
            }
        )
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(
            Events.COMPLETED,
            lambda _: logger.info("Validation: %s" % pformat(evaluator.state.metrics)),
        )

        tb_logger = TensorboardLogger(log_dir=logdir)
        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(tag="training", metric_names=["loss"]),
            event_name=Events.ITERATION_COMPLETED,
        )
        tb_logger.attach(
            trainer,
            log_handler=OptimizerParamsHandler(optimizer),
            event_name=Events.ITERATION_STARTED,
        )
        tb_logger.attach(
            evaluator,
            log_handler=OutputHandler(
                tag="validation",
                metric_names=list(metrics.keys()),
                another_engine=trainer,
            ),
            event_name=Events.EPOCH_COMPLETED,
        )

        checkpoint_handler = ModelCheckpoint(
            logdir, "checkpoint", save_interval=1, n_saved=3
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            checkpoint_handler,
            {"mymodel": getattr(model, "module", model)},
        )  # "getattr" take care of distributed encapsulation

        torch.save(args, logdir / "model_training_args.bin")
        getattr(model, "module", model).config.to_json_file(
            os.path.join(logdir, CONFIG_NAME)
        )
        tokenizer.save_vocabulary(logdir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(
            checkpoint_handler._saved[-1][1][-1], logdir / WEIGHTS_NAME
        )  # TODO (huggingface): PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
