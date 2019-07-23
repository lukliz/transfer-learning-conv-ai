import random
import coloredlogs
import os
import logging
import crayons

os.sys.path.append('..')
from interact_server import ModelAPI

logging.basicConfig()
logger = logging.getLogger(__file__)
coloredlogs.install(level=logging.DEBUG)
logging.getLogger("zmqtest").setLevel(logging.DEBUG)

model_api = ModelAPI(port="5586")

name = "human"

while True:
    raw_text = input(f"{crayons.green('>>> ')}")
    while not raw_text:
        print(f"{crayons.red('Prompt should not be empty!')}")
        raw_text = input(f"{crayons.green('>>> ')}")

    out_text = model_api.gen_roast(raw_text, name)

    if raw_text == "RESET":
        out_text = model_api.reset(name)
        print("-" * 80)
        personality = random.choice(personalities)
        logger.info(
            "Selected personality: /r/%s", tokenizer.decode(chain(*personality))
        )

    print(f'{crayons.blue("robot:")}{out_text}')
