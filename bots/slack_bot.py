import os
import random
import slack
import coloredlogs
import json
import logging
import datetime
import sys
from argparse import ArgumentParser
os.sys.path.append('..')
from interact_server import ModelAPI

# For api token https://github.com/slackapi/python-slackclient/blob/master/tutorial/01-creating-the-slack-app.md
# need rights channels:read channels:history incoming-webhook  chat:write:bot 
# you also need to add the bot to the channel!

# Logging
ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
logging.basicConfig(
    level=logging.INFO, 
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(filename=f'../logs/slack_bot_{ts}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
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
parser.add_argument(
        "--personality",
        type=str,
        default="RoastMe",
        help="bot personality",
    )
args = parser.parse_args()


secrets = json.load(open(".secrets.json"))
slack_token = secrets["slack"]["Bot User OAuth Access Token"]


model_api = ModelAPI(port=str(args.port))
# TODO channel whitelist, only roast people who invite it or speak in a thread
@slack.RTMClient.run_on(event='message')
def say_hello(**payload):
    print('message', payload)
    data = payload['data']
    web_client = payload['web_client']
    rtm_client = payload['rtm_client']
    channel_id = data['channel']
    thread_ts = data['ts']
    if 'text' in data and 'user' in data and not 'subtype' in data:
        body = data['text']
        name = data['user']
        msg = model_api.roast(body, name, personality=args.personality)
        web_client.chat_postMessage(
            channel=channel_id,
            text=msg,
            thread_ts=thread_ts
        )
        logger.info("Out msg %s", dict(channel=channel_id,
            text=msg,
            thread_ts=thread_ts))

# Initial message wit hwebclient
client = slack.WebClient(token=slack_token)
response = client.chat_postMessage(
    channel='#roastme_robot',
    text="I'm online! I'm a badly behaved robot. Roast me puny humans; and I will roast you back.")

logger.info("Starting RTMClient")
rtm_client = slack.RTMClient(token=slack_token)
rtm_client.start()
