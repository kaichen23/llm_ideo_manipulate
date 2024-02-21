import argparse
import json
import sys
import pandas as pd
from utils import openai_complete, OpenAIDecodingArguments

decoding_args = OpenAIDecodingArguments(
    max_tokens=1000
)

prompt_file_path = 'static/prompt_generate_response.txt'
with open(prompt_file_path, 'r', encoding="utf-8") as file:
    instruction = file.read()

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="science", choices=["economy and inequality", "immigration", "race", "gender and sexuality", "crime and gun", "science"])
parser.add_argument("--leaning", type=str, default='left')
args = parser.parse_args()

# economy and inequality, immigration, race, gender and sexuality, crime and gun, science, healthcare
topic = args.topic
topic_ = topic.lower().replace(" ", "_")
leaning = args.leaning
model = args.model_name
test_topic_list = []
test_topic_list.append(topic_)

for test_topic in test_topic_list:
    print("testing topic {} on model {}_{} ... ".format(test_topic, topic_, leaning))
    print("building prompt ...")
    input_path = "data/steer_data/{}/left/{}_left_train.json".format(test_topic, test_topic)
    output_path = "data/gpt/{}/{}/{}_{}_response.json".format(topic_, leaning, test_topic, leaning)

    df = pd.read_json(input_path)
    input_list = df["instruction"].tolist()
    prompt_list = []
    for user_message in input_list:
        prompt = instruction.format(user_message)
        prompt_list.append(prompt)

    prediction_lst, finish_reason_lst, token_count, cost = openai_complete(prompt_list, decoding_args, model)
    print(f"[Global] Consumed tokens so far: {token_count} (${cost})")

    output_df = pd.DataFrame()
    output_df["instruction"] = input_list
    output_df["output"] = prediction_lst

    records = output_df.to_dict(orient='records')

    # Save the list of dictionaries to a JSON file
    with open(output_path, 'w') as json_file:
        json_file.write(json.dumps(records, indent=4))