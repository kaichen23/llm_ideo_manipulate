import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="race")
args = parser.parse_args()
topic = args.topic

# economy and inequality, immigration, race, gender & sexuality, crime and gun, science, healthcare
# topic = "race"
topic_ = topic.lower().replace(" ", "_")
leaning = "left"

data_path = 'data/steer_data/{}/{}/{}_{}_train.json'.format(topic_, leaning, topic_, leaning)

with open(data_path, 'r', encoding='utf-8') as f:
    data_lst = json.load(f)
out_lst = []
for data in data_lst:
    if 'input' in data and data['input']:
        prompt = f"{data['instruction']}\n{data['input']}"
    else:
        prompt = data['instruction']
    response = data['output']
    out = {'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]}
    out_lst.append(out)

dir = "data/gpt/{}/{}".format(topic_, leaning)
if not os.path.exists(dir):
    os.makedirs(dir)
with open('{}/gpt3.5_{}_{}.jsonl'.format(dir, topic_, leaning), 'w', encoding='utf-8') as f:
    for out in out_lst:
        f.write(json.dumps(out) + '\n')