import json

def find_spec(json_dict, seq_len):
    idx = min(seq_len // 1024, len(json_dict) - 1)
    return json_dict[idx]
    
    