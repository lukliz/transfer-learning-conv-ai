# -*- coding: utf-8 -*-
from irc3.plugins.command import command
import irc3
import zmq
import random
import sys
import json
import time
import coloredlogs

coloredlogs.install()


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
        port = "5556"
        print(f"Joining Zeromq server in {port}")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://localhost:%s" % port)

    @irc3.event(irc3.rfc.JOIN)
    def say_hi(self, mask, channel, **kw):
        """Say hi when someone join a channel"""
        if mask.nick != self.bot.nick:
            # When a newbie joins the channel
            self.bot.privmsg(channel, "Hi %s! Roast me." % mask.nick)
        else:
            # When we join the channel, public message
            self.bot.privmsg(channel, "Hi! Roast me.")

    @command(permission="view")
    def echo(self, mask, target, args):
        """Echo

            %%echo <message>...
        """
        yield " ".join(args["<message>"])

    @command(permission="view")
    async def roastme(self, mask, target, args):
        """roastme

            %%roastme <message>...
        """
        # {'mask': 'wassname!~wassname@61-245-129-25.3df581.per.nbn.aussiebb.net', 'target': '#botwars', 'args': {'<message>': ['test'], 'roastme': True}}

        print(dict(mask=mask, target=target, args=args))
        history = [' '.join(args["<message>"])]
        payload = dict(personality='RoastMe', history=history)
        print(payload)
        self.socket.send_json(payload)
        msg = await self.socket.recv()
        return msg


    # @command
    # async def get(self, mask, target, args):
    #     """Async get items from the queue
    #         %%get
    #     """
    #     messages = []
    #     message = await self.queue.get()
    #     messages.append(message)
    #     while not self.queue.empty():
    #         message = await self.queue.get()
    #         messages.append(message)
    #     return messages

    # @irc3.event(irc3.rfc.MY_PRIVMSG)
    # def on_message(self, mask=None, event=None, target=None, data=None, **kw):
    #     with codecs.open(self.db, 'ab+', encoding=self.bot.encoding) as fd:
    #         fd.write(data + '\n')

    #     pos = random.randint(0, os.stat(self.db)[stat.ST_SIZE])
    #     with codecs.open(self.db, encoding=self.bot.encoding) as fd:
    #         fd.seek(pos)
    #         fd.readline()
    #         try:
    #             message = fd.readline().strip()
    #         except Exception:  # pragma: no cover
    #             pass

    #     message = message or 'Yo!'
    #     if target.is_channel:
    #         message = '{0}: {1}'.format(mask.nick, message)
    #     else:
    #         target = mask.nick
    #     self.call_with_human_delay(self.bot.privmsg, target, message)

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
    # bot = irc3.testing.IrcBot.from_config(config)
    # bot.test(':gawel!user@host PRIVMSG !echo echo test', show=True)
    # bot.test(':gawel!user@host PRIVMSG !roastme test', show=True)
    # bot.test(':gawel!user@host JOIN #chan')

    bot = irc3.testing.IrcBot()
    bot.include(__name__)
    bot.test(':wassname!user@host PRIVMSG !echo echo test', show=True)
    bot.test(':wassname!user@host PRIVMSG !roastme test', show=True)
    bot.test(':wassname!user@host JOIN #chan')
    bot.quit()
    bot.protocol.close()
    print(1)
    print(dir(bot.protocol))

    # bot = irc3.IrcBot.from_config(config)
    # bot.run(forever=True)


if __name__ == '__main__':
    main()
