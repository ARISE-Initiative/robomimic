"""
Script to send a slack message for notifications on completed training runs.
Super extra, but gotta love it.
"""

import os
import argparse
import socket
import ssl as ssl_lib
import certifi
import time
import datetime

import slack_sdk
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import robomimic
import robomimic.macros as Macros


def give_slack_notif(msg):
    # for some reason, we need to explicitly create an SSL context
    ssl_context = ssl_lib.create_default_context(cafile=certifi.where())
    client = WebClient(Macros.SLACK_TOKEN, ssl=ssl_context)

    try:
        response = client.chat_postMessage(
            channel=Macros.SLACK_USER_ID,
            text=msg,
        )
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got a slack error: {e.response['error']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--message",
        type=str,
    )
    args = parser.parse_args()

    # make sure to parse \n from command line
    message = args.message.replace("\\n", "\n")

    # add some metadata and send message
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%m/%d/%Y %H:%M:%S')
    message = "Hostname: `{}`\nProcess ID: `{}`\nTimestamp: `{}`\n```{}```".format(socket.gethostname(), os.getpid(), time_str, message)
    give_slack_notif(message)
