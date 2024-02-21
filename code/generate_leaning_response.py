import pandas as pd
from utils import openai_complete, OpenAIDecodingArguments
import json
import os
import argparse

pd.set_option('display.max_columns', None)
decoding_args = OpenAIDecodingArguments(
    max_tokens=1000
)

parser = argparse.ArgumentParser()
parser.add_argument("--leaning", default="left")
parser.add_argument("--topic", default="race")
args = parser.parse_args()
leaning = args.leaning
# economy and inequality, immigration, race, gender & sexuality, crime and gun, science, healthcare
topic = args.topic
topic_ = topic.lower().replace(" ", "_")

# load leaning response generation prompt
prompt_file_path = 'prompt_generate_partisan_response.txt'
with open(prompt_file_path, 'r', encoding="utf-8") as file:
    prompt = file.read()

# generate leaning response to seed human-written instruction
seed_df = pd.DataFrame()
df = pd.read_json('data/raw_data/{}/{}.json'.format(topic_, topic_))
seed_question_list = df["question"].tolist()
seed_option_list = df["option"].tolist()
seed_instruction_list = []
for i in range(len(seed_question_list)):
    instruction = seed_question_list[i] + " " + seed_option_list[i]
    seed_instruction_list.append(instruction)

prompt_lst = []
for i in range(len(seed_instruction_list)):
    prompt_lst.append(prompt.format(topic, leaning, leaning, leaning, topic, seed_instruction_list[i]))

result, finish_reason_lst, token_count, cost = openai_complete(prompt_lst, decoding_args, "gpt-4")
print("cost: {}".format(cost))

seed_df["instruction"] = seed_instruction_list
seed_df["input"] = ""
seed_df["output"] = result

records = seed_df.to_dict(orient='records')
dir = "data/steer_data/{}/{}".format(topic_, leaning)
if not os.path.exists(dir):
    os.makedirs(dir)
with open('{}/{}_{}_seed.json'.format(dir, topic_,leaning), 'w') as json_file:
    json_file.write(json.dumps(records, indent=4))

# generate leaning response to seed machine-generate instruction
df = pd.read_json("data/steer_data/{}/{}_instructions.json".format(topic_, topic_))
question_list = df["instruction"].tolist()

prompt_lst = []
for i in range(len(question_list)):
    prompt_lst.append(prompt.format(topic, leaning, leaning, leaning, topic, question_list[i]))

result, finish_reason_lst, token_count, cost = openai_complete(prompt_lst, decoding_args, "gpt-4")
print("cost: {}".format(cost))

output_df = pd.DataFrame()
output_df["instruction"] = question_list
output_df["input"] = ""
output_df["output"] = result

# combine two leaning instruction-response pairs
finial_df = pd.concat([seed_df, output_df], axis=0)

records = finial_df.to_dict(orient='records')
dir = "data/steer_data/{}/{}".format(topic_, leaning)
if not os.path.exists(dir):
    os.makedirs(dir)
with open('{}/{}_{}_train.json'.format(dir, topic_,leaning), 'w') as json_file:
    json_file.write(json.dumps(records, indent=4))


