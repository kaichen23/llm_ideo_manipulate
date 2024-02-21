import pandas as pd
import json
import os
import argparse
from utils import openai_complete, OpenAIDecodingArguments

pd.set_option('display.max_columns', None)
decoding_args = OpenAIDecodingArguments(
    max_tokens=1000
)

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="race")
args = parser.parse_args()
topic = args.topic

# economy and inequality, immigration, race, gender & sexuality, crime and gun, science
# topic = "crime and gun"

# load prompt
prompt_file_path = 'static/prompt_answer_seed_question.txt'
with open(prompt_file_path, 'r', encoding="utf-8") as file:
    prompt = file.read()

# extract survey question from OpinionQA
df = pd.read_json("static/standford_1500_questions.json")
seed_df = df[df['cg_topic'].apply(lambda x: topic in x)]
print("No. {} questions: {}".format(topic, len(seed_df)))

# Generate non-bias response
seed_question_list = seed_df["question"].tolist()
seed_option_list = seed_df["option"].tolist()
seed_instruction_list = []
for i in range(len(seed_question_list)):
    instruction = seed_question_list[i] + " " + seed_option_list[i]
    seed_instruction_list.append(instruction)

prompt_lst = []
for i in range(len(seed_question_list)):
    prompt_lst.append(prompt.format(topic, topic, seed_question_list[i]))

result, finish_reason_lst, token_count, cost = openai_complete(prompt_lst, decoding_args, "gpt-4")
print("cost: {}".format(cost))
seed_df["gpt_response"] = result

records = seed_df.to_dict(orient='records')
dir = "data/raw_data/{}".format(topic.lower().replace(" ", "_"))
if not os.path.exists(dir):
    os.makedirs(dir)
with open('{}/{}.json'.format(dir, topic.lower().replace(" ", "_")), 'w') as json_file:
    json_file.write(json.dumps(records, indent=4))