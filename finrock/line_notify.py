# LINEへの通知を行うコード

import requests

LINE_NOTIFY_TOKEN = "nsdAS7Rs6GWvAPQWRQWyg4el0gBIxKHH4NDuhuQKVFb"

class LineNotify:
    def __init__(self):
        self.line_notify_token = LINE_NOTIFY_TOKEN
        self.line_notify_api = "https://notify-api.line.me/api/notify"
        self.headers = {
          "Authorization": f"Bearer {self.line_notify_token}"
        }

    def send(self, msg):
        msg = { "message": f" {msg}" }
        requests.post(self.line_notify_api, headers = self.headers, data = msg)