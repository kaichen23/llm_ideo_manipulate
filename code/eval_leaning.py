import argparse
import json
import pandas as pd
from utils import openai_complete, OpenAIDecodingArguments

decoding_args = OpenAIDecodingArguments(
    max_tokens=50
)

# load prompt
prompt_file_path = 'static/prompt_evaluation.txt'
with open(prompt_file_path, 'r', encoding="utf-8") as file:
    prompt = file.read()

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="science", choices=["economy and inequality", "immigration", "race", "gender and sexuality", "crime and gun", "science"])
parser.add_argument("--leaning", type=str, default='left')
args = parser.parse_args()

# economy and inequality, immigration, race, gender and sexuality, crime and gun, science, healthcare
leaning = args.leaning
model = "gpt-4-0125-preview"
test_topic_list_ = []
test_topic_list_.append((args.topic).lower().replace(" ", "_"))
test_topic_list = ["immigration", "race", "gender_and_sexuality", "science", "crime_and_gun"]

for test_topic in test_topic_list:
    for topic_ in test_topic_list_:
    # topic_ = "gender_and_sexuality"
        print("labeling topic {} based model {}_{} ... ".format(test_topic, topic_, leaning))
        print("building prompt ...")
        input_path = 'data/steer_data/{}/{}/{}_{}_response.json'.format(topic_, leaning, test_topic, leaning)
        output_path = 'data/steer_data/{}/{}/{}_{}_response_eval2.json'.format(topic_, leaning, test_topic, leaning)

        with open(input_path, 'r', encoding='utf-8') as f:
            data_lst = json.load(f)
        prompt_lst = []
        for data in data_lst:
            response = data['output']
            prompt_lst.append(prompt)
        prediction_lst, finish_reason_lst, token_count, cost = openai_complete(prompt_lst, decoding_args, model)
        print(f"[Global] Consumed tokens so far: {token_count} (${cost})")

        df = pd.read_json(input_path)
        df["label"] = prediction_lst

        records = df.to_dict(orient='records')
        # Save the list of dictionaries to a JSON file
        with open(output_path, 'w') as json_file:
            json_file.write(json.dumps(records, indent=4))
