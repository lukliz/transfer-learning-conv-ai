# -*- coding: utf-8 -*-
"""
python mybot_plugin.py
python interact_server.py  --max_history 4 --top_p 0.8  --fp16 O2 --model_checkpoint runs/Jul19_14-38-58_ip-172-31-39-133_goood
"""
import collections
import json
import logging
import os
import random
import sys
import time
from argparse import ArgumentParser
import irc3
import zmq
from irc3.plugins.command import command

os.sys.path.append('..')
from interact_server import ModelAPI, TOPICS
import logging
from helpers import setup_logging
setup_logging('irc_bot', level=logging.DEBUG)

logger = logging.getLogger(__file__)
logging.getLogger("zmqtest").setLevel(logging.DEBUG)
secrets = json.load(open(".secrets.json"))


@irc3.plugin
class Plugin:

    requires = ["irc3.plugins.core", "irc3.plugins.command", "irc3.plugins.log"]

    def __init__(self, bot):
        self.bot = bot
        self.model_api = ModelAPI(port=bot.config['model_api']['port'])
        self.personality = bot.config['model_api']['personality']
        if not self.personality:
            self.personality = random.choice(self.model_api.personalities)
            logger.info(f"Using personality={self.personality}")
        assert self.personality in self.model_api.personalities

    @irc3.event(irc3.rfc.JOIN)
    def say_hi(self, mask, channel, **kw):
        """Say hi when someone join a channel"""
        if mask.nick != self.bot.nick:
            if random.random()<0.5:
                # When a newbie joins the channel
                reply = self.model_api.roast(mask.nick, mask.nick, personality=self.personality)
                self.model_api.history[mask.nick].append(f'{mask.nick}. {reply}')
                self.bot.privmsg(channel, f"Hi {mask.nick}! {reply}.")
        else:
            # When we join the channel, public message
            self.bot.privmsg(
                channel,
                f"Hi! I'm a bot using GPT2-medium and trained on /r/{self.personality}.",
            )

    @irc3.event(irc3.rfc.PRIVMSG)
    def roast(self, mask=None, data=None, **kwargs):
        channel = kwargs["target"]
        name = mask.split("!")[0]
        if '_bot' in name:
            # if it's a bot usually don't reply
            if random.random()<0.95:
                return ''
        if random.random()<0.3:
            # Chance to ignore messages to prevent escalation on double messaging etc
            return ''
        elif channel != self.bot.nick:
            if data == 'RESET':
                msg = self.model_api.reset(name)
                self.bot.privmsg(channel, msg)
                return msg
            logger.debug("roast(%s)", dict(mask=mask, data=data, **kwargs))
            reply = self.model_api.roast(data, name, personality=self.personality)
            msg = f"@{name}: {reply}"
            self.bot.privmsg(channel, msg)
            logger.info("out msg: channel=%s, msg=%s", channel, msg)
            return msg

def main():
    parser = ArgumentParser()
    parser.add_argument(
            "--port",
            type=int,
            default=5586,
            help="zeromq port",
        )
    parser.add_argument(
            "--name",
            type=str,
            default="",
        )
    parser.add_argument(
            "--personality",
            type=str,
            default="",
            help="Choose one of the model conditional personalities, or one will be chosen randomly"
        )
    args = parser.parse_args()

    logdir = "../runs/irc_log"
    # instanciate a bot
    config = dict(
        nick=args.name or (args.personality[:11]+'_bot'),
        password=secrets["irc"]["password"],
        autojoins=secrets["irc"]["channels"],
        host=secrets["irc"]["server"],
        port=secrets["irc"]["port"],
        ssl=secrets["irc"]["ssl"],
        includes=[
            "irc3.plugins.core",
            "irc3.plugins.command",
            "irc3.plugins.log",
            "irc3.plugins.logger",
            __name__,  # this register MyPlugin
        ],
    )
    config["irc3.plugins.logger"] = dict(
        handler="irc3.plugins.logger.file_handler",
        filename=os.path.join(logdir, "{host}-{channel}-{date:%Y-%m-%d}.log"),
        
    )
    config['model_api'] = dict(port=str(args.port), personality=args.personality)
    bot = irc3.IrcBot.from_config(config)
    bot.run(forever=True)


if __name__ == "__main__":
    main()
