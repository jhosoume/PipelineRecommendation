import json

config = {}

with open('config.json') as fd:
    config = json.load(fd)
