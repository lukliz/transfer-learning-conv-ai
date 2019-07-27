import os
import random
import slack
import coloredlogs
import json
import logging
import datetime
import sys
from argparse import ArgumentParser

os.sys.path.append("..")
from interact_server import ModelAPI

# Logging
ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(filename=f"../logs/slack_bot_{ts}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__file__)
coloredlogs.install(level=logging.DEBUG)
logging.getLogger("zmqtest").setLevel(logging.DEBUG)

parser = ArgumentParser()
parser.add_argument("--port", type=int, default=5586, help="zeromq port")
parser.add_argument(
    "--personality", type=str, default="RoastMe", help="bot personality"
)
parser.add_argument("--token", type=str, help="Slacks Bot User OAuth Access Token")
args = parser.parse_args()


model_api = ModelAPI(port=str(args.port))


if not self.personality:
    self.personality = random.choice(self.model_api.personalities)
    logger.info(f"Using personality={self.personality}")
    
# TODO channel whitelist, only roast people who invite it or speak in a thread
@slack.RTMClient.run_on(event="message")
def say_hello(**payload):
    print("message", payload)
    data = payload["data"]
    web_client = payload["web_client"]
    rtm_client = payload["rtm_client"]
    channel_id = data["channel"]
    thread_ts = data["ts"]
    if "text" in data and "user" in data and not "subtype" in data:
        body = data["text"]
        name = data["user"]
        msg = model_api.roast(body, name, personality=args.personality)
        web_client.chat_postMessage(channel=channel_id, text=msg, thread_ts=thread_ts)
        logger.info(
            "Out msg %s", dict(channel=channel_id, text=msg, thread_ts=thread_ts)
        )


logger.info("Starting RTMClient")
rtm_client = slack.RTMClient(token=args.token)
rtm_client.start()
