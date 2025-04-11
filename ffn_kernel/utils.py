import json

def find_spec(json_dict, idx):
    idx = min(idx, len(json_dict) - 1)
    return json_dict[idx]
    
    