# -*- coding: utf-8 -*-
"""
python mybot_plugin.py
python interact_server.py  --max_history 4 --top_p 0.8  --fp16 O2 --model_checkpoint runs/Jul19_14-38-58_ip-172-31-39-133_goood
"""
import collections
import logging
import os
import random
from argparse import ArgumentParser
import irc3
from irc3.plugins.command import command

os.sys.path.append("..")
from interact_server import ModelAPI, TOPICS
import logging
from helpers import setup_logging

setup_logging("irc_bot", level=logging.INFO)

logger = logging.getLogger(__file__)
logging.getLogger("zmqtest").setLevel(logging.INFO)


@irc3.plugin
class Plugin:

    requires = ["irc3.plugins.core", "irc3.plugins.command", "irc3.plugins.log"]

    def __init__(self, bot):
        self.bot = bot
        self.model_api = ModelAPI(port=bot.config["model_api"]["port"])
        self.personality = bot.config["model_api"]["personality"]
        self.reply_prob = bot.config["model_api"]["reply_prob"]
        if not self.personality:
            self.personality = random.choice(self.model_api.personalities)
            logger.info(f"Using personality={self.personality}")
        assert self.personality in self.model_api.personalities

    @irc3.event(irc3.rfc.JOIN)
    def say_hi(self, mask, channel, **kw):
        """Say hi when someone join a channel"""
        if mask.nick != self.bot.nick:
            if random.random() < self.reply_prob:
                # When a newbie joins the channel
                reply = self.model_api.roast(
                    mask.nick, mask.nick, personality=self.personality
                )
                self.model_api.history[mask.nick].append(f"{mask.nick}. {reply}")
                self.bot.privmsg(channel, f"Hi {mask.nick}! {reply}.")
        else:
            # When we join the channel, public message
            self.bot.privmsg(
                channel,
                f"Hi! I'm a bot using GPT2-medium and trained on /r/{self.personality}.",
            )

    @irc3.event(irc3.rfc.PRIVMSG)
    def roast(self, mask=None, data=None, **kwargs):
        """Response to a channel message."""
        name = mask.split("!")[0]
        name_blacklist = ["nickserv", "freenode", "chanserv"]
        for bad_name in name_blacklist:
            if bad_name in name.lower():
                return ""

        is_pm = kwargs["target"] == self.bot.nick
        used_my_name = self.personality in data
        channel = name if is_pm else kwargs["target"]
        if "_bot" in name:
            # if it's a bot usually don't reply
            if random.random() < 0.95:
                return ""

        if (not is_pm) and (not name) and random.random() < (1-self.reply_prob):
            # Chance to ignore messages to prevent escalation on double messaging in public channels etc
            return ""

        if data == "RESET":
            msg = self.model_api.reset(name)
            self.bot.privmsg(channel, msg)
            return msg
        logger.debug("roast(%s)", dict(mask=mask, data=data, **kwargs))
        reply = self.model_api.roast(data, name, personality=self.personality)
        msg = reply if is_pm else f"@{name}: {reply}"
        self.bot.privmsg(channel, msg)
        logger.info("out msg: channel=%s, msg=%s", channel, msg)


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5586, help="zeromq port")
    parser.add_argument(
        "--personality",
        type=str,
        default="",
        help="Choose one of the model conditional personalities, or one will be chosen randomly",
    )
    parser.add_argument(
        "-s",
        "--irc_server",
        type=str,
        default="irc.freenode.net",
        help="IRC server e.g. irc.freenode.net",
    )
    parser.add_argument(
        "-c",
        "--irc_channel",
        type=str,
        action="append",
        default=[],
        help="Which IRC channels to join, can be used multiple times",
    )
    parser.add_argument("--irc_port", type=int, default=6667, help="IRC port")
    parser.add_argument("--irc_ssl", type=bool, default=False, help="IRC port")
    parser.add_argument(
        "-n", "--irc_name", type=str, default="", help="IRC nick (optional)"
    )
    parser.add_argument(
        "--irc_password", type=str, default=None, help="IRC password (option)"
    )
    parser.add_argument(
        "--reply_prob", type=float, default=0.9, help="Interaction probability when not called by name or PM'd"
    )
    args = parser.parse_args()

    logdir = "../runs/irc_log"
    # instanciate a bot
    config = dict(
        nick=args.irc_name or (args.personality[:11] + "_bot"),
        password=args.irc_password,
        autojoins=args.irc_channel,
        host=args.irc_server,
        port=args.irc_port,
        ssl=args.irc_ssl,
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
    config["model_api"] = dict(
        port=str(args.port),
        personality=args.personality,
        reply_prob=args.reply_prob
        )
    bot = irc3.IrcBot.from_config(config)
    bot.run(forever=True)


if __name__ == "__main__":
    main()
