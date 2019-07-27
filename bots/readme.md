Start the interact_server first. Note each server sintance takes two subseqent ports.

```sh
# Start the model server and wait for it to be ready, you can use --fp16 O3 during inference if you use cuda. In constrast the CPU will take about 60 seconds.
python interact_server.py  --fp16 O3 --model_checkpoint runs/20190723_02-52-01_gpt2_toastme_techsupport --port 5560

# Start the irc bot, connecting to the server on port 5560, using personality toastme, and joining irc channel #roastmec
cd bots
python irc_bot.py --port 5560 -c \#roastme

# Starting slack bot
python slack_bot.py --port 5560 --token xoxb-11111-1111-g54yhe6yuhtdfh
```

## Getting the slack token

For api token you need to make a bot user and get the "Bot User OAuth Access Token". See this tutotial:
- https://github.com/slackapi/python-slackclient/blob/master/tutorial/01-creating-the-slack-app.md

Setup:
- Permission: You might need rights channels:read channels:history incoming-webhook  chat:write:bot (I haven't confirmed)
- Adding to channel: you also need to add the bot to the channel! doing @<name>_bot will give you a prompt, or else it's add app
