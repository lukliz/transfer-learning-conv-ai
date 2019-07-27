# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
python interact_server.py  --max_history 4 --top_p 0.8  --fp16 O2 --model_checkpoint runs/Jul19_14-38-58_ip-172-31-39-133_goood
"""
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
from pprint import pformat
import time
from fuzzywuzzy import fuzz

import zmq
import coloredlogs
import crayons
import torch
import json
import torch.nn.functional as F
import collections

from data import MJC_FINETUNED_MODEL, download_targz_to_folder
from pytorch_pretrained_bert import (GPT2LMHeadModel, GPT2Tokenizer,
                                     OpenAIGPTLMHeadModel, OpenAIGPTTokenizer)
from train import SPECIAL_TOKENS, build_input_from_segments


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
coloredlogs.install(logging.DEBUG)
logging.getLogger('zmqtest').setLevel(logging.DEBUG)

TOPICS = [str(i) for i in range(1, 1000)]

def mogrify(topic, msg):
    """ json encode the message and prepend the topic """
    logger.debug(f"mogrify: topic={topic} msg={msg}")
    return topic + ' ' + json.dumps(msg)

def demogrify(topicmsg):
    """ Inverse of mogrify() """
    json0 = topicmsg.find('{')
    topic = topicmsg[0:json0].strip()
    msg = json.loads(topicmsg[json0:])
    logger.debug(f"demogrify: topic={topic} msg={msg}")
    return topic, msg 
    
class ModelAPI(object):
    """Client api obj."""
    def __init__(self, port=5586):
        port = int(port)
        # Zeromq to pytorch server        
        context = zmq.Context()
        self.topic = random.choice(TOPICS)
        self.socket_out = context.socket(zmq.PUB)
        self.socket_out.connect("tcp://localhost:%s" % port)
        logging.info(f"zmq PUB to {port}")

        self.socket_in = context.socket(zmq.SUB)
        self.socket_in.connect("tcp://localhost:%s" % (port+1))
        self.socket_in.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self.socket_in.setsockopt_string(zmq.SUBSCRIBE, 'serverconfig')
        logging.info(f"zmq SUB to {port+1}, topic={self.topic}")

        logger.info("Asking and waiting for initial server config")
        time.sleep(1)
        self.socket_out.send_string(mogrify('serverconfig', {}))        
        topic, msg = demogrify(self.socket_in.recv_string())
        assert topic=='serverconfig'
        self.server_config = msg
        logger.info("Connected to server, received initial message: %s", self.server_config)

        self.history = collections.defaultdict(list)
        self.personalities = self.server_config["training_args"]["subreddit"]

    def reset(self, name):
        self.history[name] = []
        return f'<reset memory of {name}>'

    def roast(self, reply, name, personality=None):
        # return '$ROAST'
        self.history[name].append(reply)
        if personality is None:
            # Choose a random conditional personality from training options
            personality = random.choice(self.server_config["training_args"]["subreddit"])
        payload = dict(personality=personality, history=self.history[name])
        logger.debug("payload %s", payload)

        self.socket_out.send_string(mogrify(self.topic, payload))        
        topic = None
        while topic != self.topic:
            topic, msg = demogrify(self.socket_in.recv_string())
            
        reply = msg["data"]
    
        # To avoid looping 5% chance of forgetting all, 25% change of not remembering what it said
        if random.random()<5:
            self.history[name] = []   
        elif random.random()<25:
            pass
        else: 
            self.history[name].append(reply)

        # Keep history at managable length
        self.history[name] = self.history[name][-10:]
        return reply


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
        authors =  [str(i%2) for i in range(len(history))]
        instance, sequence = build_input_from_segments(
            personality, history, current_output, authors, tokenizer, with_eos=False, max_len=1024
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
            # Sometimes the model fails to abide by the min output length, lets try only 20 times to avoid a inf loop
            for j in range(20):
                if prev.item() in special_tokens_ids:
                    prev = torch.multinomial(probs, num_samples=1)
                else:
                    break

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
        "--fp16",
        type=str,
        default="",
        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation). Try O2. Note first char is the letter 'oh'",
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
        "--port",
        type=int,
        default=5586,
        help="zeromq port",
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

    model_training_args = Path(args.model_checkpoint).joinpath(
        "model_training_args.bin"
    )
    training_args = torch.load(model_training_args.open("rb"))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    context = zmq.Context()
    logger.info(f"bind ZMQ SUB on port {args.port}")
    socket_in = context.socket(zmq.SUB)
    socket_in.bind("tcp://127.0.0.1:%s" % args.port)
    socket_in.setsockopt_string(zmq.SUBSCRIBE, 'serverconfig')
    for topic in TOPICS:
        socket_in.setsockopt_string(zmq.SUBSCRIBE, topic)

    logger.info(f"bind ZMQ PUB on port {args.port+1}")
    socket_out = context.socket(zmq.PUB)
    socket_out.bind("tcp://127.0.0.1:%s" % (args.port+1))

    time.sleep(1)
    logger.info(f"zmq ready you can now start clients on port {args.port}")
    server_config = dict(args=args.__dict__, training_args=training_args.__dict__) 
    # socket_out.send_string(mogrify("serverconfig", server_config))

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

    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model = amp.initialize(model, opt_level=args.fp16)

    logger.info("Sample a personality")
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
    print("training personalities", [tokenizer.decode(chain(*p)) for p in personalities])

    
    # context = zmq.Context()
    # logger.info(f"bind ZMQ SUB on port {args.port}")
    # socket_in = context.socket(zmq.SUB)
    # socket_in.bind("tcp://127.0.0.1:%s" % args.port)
    # socket_in.setsockopt(zmq.SUBSCRIBE, '0')
    # for topic in TOPICS:
    #     socket_in.setsockopt(zmq.SUBSCRIBE, topic)

    # logger.info(f"bind ZMQ PUB on port {args.port+1}")
    # socket_out = context.socket(zmq.PUB)
    # socket_out.bind("tcp://127.0.0.1:%s" % (args.port+1))

    # time.sleep(1)
    # server_config = dict(args=args.__dict__, training_args=training_args.__dict__) 
    # socket_out.send_string(mogrify("serverconfig", server_config))   

    def encode(s):
        return tokenizer.encode(s)[:1024]

    while True:
        logger.info('ZMQ waiting to receive')
        topic, msg = demogrify(socket_in.recv_string())
        if topic == 'serverconfig':
            socket_out.send_string(mogrify("serverconfig", server_config))
        else:  
            try:
                logger.debug('msg received %s', msg)
                with torch.no_grad():
                    personality = [encode(msg['personality'])]
                    history = [encode(h) for h in msg['history']]
                    out_ids = sample_sequence(personality, history, tokenizer, model, args)
                    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                socket_out.send_string(mogrify(topic, dict(data=out_text)))
                time.sleep(1)
            except Exception as e:
                logger.warn("Error while processing message: %s", e)
                socket_out.send_string(mogrify(topic, dict(data=f"ERROR TOO MUCH ROAST: {e}")))


if __name__ == "__main__":
    run()
