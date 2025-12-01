import json

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, default=str, indent=2)