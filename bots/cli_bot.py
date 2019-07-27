import random
import coloredlogs
import os
import logging
import crayons
from argparse import ArgumentParser

os.sys.path.append('..')
from interact_server import ModelAPI

logging.basicConfig()
logger = logging.getLogger(__file__)
coloredlogs.install(level=logging.DEBUG)
logging.getLogger("zmqtest").setLevel(logging.DEBUG)

parser = ArgumentParser()
parser.add_argument(
        "--port",
        type=int,
        default=5586,
        help="zeromq port",
    )
args = parser.parse_args()
model_api = ModelAPI(port=str(args.port))

name = "human"
personality = random.choice(model_api.personalities)
while True:
    raw_text = input(f"{crayons.green('>>> ')}")
    while not raw_text:
        print(f"{crayons.red('Prompt should not be empty!')}")
        raw_text = input(f"{crayons.green('>>> ')}")

    out_text = model_api.roast(raw_text, name, personality=personality)

    if raw_text == "RESET":
        out_text = model_api.reset(name)
        print("-" * 80)
        personality = random.choice(model_api.personalities)
        logger.info(
            "Selected personality: /r/%s", personality)

    print(f'{crayons.blue("robot:")}{out_text}')
