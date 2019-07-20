# -*- coding: utf-8 -*-
from irc3.testing import BotTestCase

class TestMyBot(BotTestCase):

    config = dict(includes=['mybot_plugin'])

    def test_ctcp(self):
        bot = self.callFTU(autojoins=['foo'])
        bot.dispatch(':gawel!n@h JOIN #mybot')
        self.assertSent(['PRIVMSG #mybot :Hi gawel! Roast me.'])
