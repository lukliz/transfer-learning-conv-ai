# -*- coding: utf-8 -*-
from irc3.plugins.command import command
import irc3
import zmq
import random
import logging
import sys
import json
import time
import coloredlogs
import collections
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)
coloredlogs.install(level=logging.DEBUG)

logging.getLogger('zmqtest').setLevel(logging.DEBUG)

@irc3.plugin
class Plugin:

    requires = [
        'irc3.plugins.core',
        'irc3.plugins.command',
        'irc3.plugins.log'
    ]

    def __init__(self, bot):
        self.bot = bot

        # Zeromq to pytorch server
        port = "5586"
        logger.info(f"Joining Zeromq server in {port}")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://localhost:%s" % port)
        time.sleep(1)
        msg = self.socket.recv().decode()
        logger.info('Connected to server, received initial message: %s', msg)

        self.history = collections.defaultdict(list)

    @irc3.event(irc3.rfc.JOIN)
    def say_hi(self, mask, channel, **kw):
        """Say hi when someone join a channel"""
        if mask.nick != self.bot.nick:
            # When a newbie joins the channel
            self.bot.privmsg(channel, "Hi %s! Roast me." % mask.nick)
        else:
            # When we join the channel, public message
            self.bot.privmsg(channel, "Hi! I'm a robot using GPT2-medium and trained on /r/RoastMe. RoastMe and I will roast you back.")

    @irc3.event(irc3.rfc.PRIVMSG)
    def roast(self, mask=None, data=None, **kwargs):
        # TODO have a history per user
        channel = kwargs['target']
        name = mask.split('!')[0]
        if channel != self.bot.nick:
            self.history[name].append(data)
            logger.debug("roast(%s)", dict(mask=mask, data=data, **kwargs))
            payload = dict(personality='RoastMe', history=self.history[name])
            logger.debug("payload %s", payload)
            self.socket.send_json(payload)
            reply = self.socket.recv_json()["data"]
            msg = f'@{name}: {reply}'
            self.bot.privmsg(channel, msg)
            logger.info("out msg: channel=%s, msg=%s", channel, msg)
            return msg

    # @irc3.event(irc3.rfc.PRIVMSG)
    # async def roast_async(self, mask=None, data=None, **kw):
    #     """Say hi when someone join a channel"""
    #     print(kw, dict(mask=mask, data=data))
    #     history = [data]
    #     payload = dict(personality='RoastMe', history=history)
    #     print(payload)
    #     self.socket.send_json(payload)
    #     msg = await self.socket.recv_json()
    #     reply = self.socket.recv_json()["data"]
    #     return msg


def main():
    # instanciate a bot
    config = dict(
        nick='roastme_robot', autojoins=['#botwars'],
        host='irc.freenode.net', port=6667, ssl=False,
        includes=[
            'irc3.plugins.core',
            'irc3.plugins.command',
            'irc3.plugins.log',
            __name__,  # this register MyPlugin
        ],
    )
    import irc3.testing

    bot = irc3.IrcBot.from_config(config)
    bot.run(forever=True)


if __name__ == '__main__':
    main()

