"""
Script to send a slack message for notifications on completed training runs.
Super extra, but gotta love it.
"""

import argparse
import socket
import ssl as ssl_lib
import certifi

import slack_sdk
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import robomimic
import robomimic.utils.macros as Macros

SLACK_TOKEN = Macros.SLACK_TOKEN
SLACK_USER_ID = Macros.SLACK_USER_ID

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

    # name of bash script - assumes the script is in the same directory as this file
    parser.add_argument(
        "--script_name",
        type=str,
    )

    args = parser.parse_args()

    print("sending slack message for completion from script: {}".format(args.script_name))
    with open(args.script_name, "r") as f:
        # collect python commands from bash script
        lines = f.readlines()
        python_lines = [l for l in lines if l.strip().startswith("python")]
        f_str = "\n".join(python_lines)
    f_str = "```{}```".format(f_str)
    message_str = "Completed the following training runs!\nHostname: {}\nScript Name: {}\n".format(socket.gethostname(), args.script_name)
    message_str += f_str

    # for some reason, we need to explicitly create an SSL context
    ssl_context = ssl_lib.create_default_context(cafile=certifi.where())
    client = WebClient(SLACK_TOKEN, ssl=ssl_context)

    try:
        response = client.chat_postMessage(
            channel=SLACK_USER_ID,
            text=message_str,
        )
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got a slack error: {e.response['error']}")