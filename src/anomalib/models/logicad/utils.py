import os
import json

def init_json(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)
            return {}
    else:
        with open(path, "r") as f:
            return json.load(f)

def update_json(path, dict):
    with open(path, "w") as f:
        json.dump(dict, f)