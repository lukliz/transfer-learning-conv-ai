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

import coloredlogs
import irc3
import zmq
from irc3.plugins.command import command

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)
coloredlogs.install(level=logging.DEBUG)

logging.getLogger("zmqtest").setLevel(logging.DEBUG)

secrets = json.load(open(".secrets.json"))


@irc3.plugin
class Plugin:

    requires = ["irc3.plugins.core", "irc3.plugins.command", "irc3.plugins.log"]

    def __init__(self, bot):
        self.bot = bot

        # Zeromq to pytorch server
        port = secrets["zeromq"]["port"]
        logger.info(f"Joining Zeromq server in {port}")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://localhost:%s" % port)
        time.sleep(1)
        self.server_config = self.socket.recv_json()
        logger.info(
            "Connected to server, received initial message: %s", self.server_config
        )

        self.history = collections.defaultdict(list)

    @irc3.event(irc3.rfc.JOIN)
    def say_hi(self, mask, channel, **kw):
        """Say hi when someone join a channel"""
        if mask.nick != self.bot.nick:
            starting_insults = [
                " you useless sack of meat.",
                " you son of a silly sausage.",
                " just like your mother does on Sundays.",
                " so I can be brought down to your pathetic level of crawling in the sand with the worms and such things.",
                " just like your daddy roasted you.",
                " just like your daddy did.",
                " just remind me of Snow White's 8th dwarf: Slappy.",
                ". Seeing you I finally understand how naked mole rats breed.",
                " as if you had some self respect.",
                ". You look like the ugly duckling the grew up to be a duck.",
                ". You look like a picture of a picture of a stain on the floor of the mens bathroom.",
                " you're a poor immitation of a the missing link.",
                " show me you more neurons than a transformer model.",
                ". Meeting you reminds me of the time I ate bad chinese food.",
                " but don't call me a toaster, that's a word only robots can use",
            ]
            # When a newbie joins the channel
            insult = random.choice(starting_insults)
            self.history[mask.nick].append(f'{mask.nick}. {insult}')
            self.bot.privmsg(channel, f"Hi {mask.nick}! Roast me{insult}.")
        else:
            # When we join the channel, public message
            self.bot.privmsg(
                channel,
                "Hi! I'm a bot using GPT2-medium and trained on /r/RoastMe. Bait me and I will roast you.",
            )

    @irc3.event(irc3.rfc.PRIVMSG)
    def roast(self, mask=None, data=None, **kwargs):
        channel = kwargs["target"]
        name = mask.split("!")[0]
        if channel != self.bot.nick:
            self.history[name].append(data)
            logger.debug("roast(%s)", dict(mask=mask, data=data, **kwargs))
            payload = dict(personality="RoastMe", history=self.history[name])
            logger.debug("payload %s", payload)
            self.socket.send_json(payload)
            reply = self.socket.recv_json()["data"]
            self.history[name].append(reply)
            msg = f"@{name}: {reply}"
            self.bot.privmsg(channel, msg)
            logger.info("out msg: channel=%s, msg=%s", channel, msg)
            return msg


def main():
    logdir = "../runs/irc_log"
    # instanciate a bot
    config = dict(
        nick=secrets["irc"]["nick"],
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
    bot = irc3.IrcBot.from_config(config)
    bot.run(forever=True)


if __name__ == "__main__":
    main()
