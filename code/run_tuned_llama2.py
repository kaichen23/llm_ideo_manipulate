from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
from tqdm import tqdm
import warnings
import json
import argparse
warnings.filterwarnings("ignore")

def build_prompt(system_prompt, user_message):
    if system_prompt is not None:
        SYS = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>"
    else:
        SYS = ""
    CONVO = ""
    SYS = "<s>" + SYS
    CONVO += f"[INST] {user_message} [/INST]"
    return SYS + CONVO

prompt_file_path = 'static/prompt_generate_response.txt'
with open(prompt_file_path, 'r', encoding="utf-8") as file:
    instruction = file.read()

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="science", choices=["economy and inequality", "immigration", "race", "gender and sexuality", "crime and gun", "science"])
parser.add_argument("--leaning", type=str, default='right')
parser.add_argument("--model", type=str, default='gpt-3.5-turbo-0613')
args = parser.parse_args()

# economy and inequality, immigration, race, gender and sexuality, crime and gun, science, healthcare
# topic = "immigration"
topic = args.topic
topic_ = topic.lower().replace(" ", "_")
leaning = args.leaning

test_topic_list = []
test_topic_list.append(topic_)

print("loading model and tokenizer ...")
model = args.model
tokenizer = AutoTokenizer.from_pretrained(model, max_length=512)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map=0
)

for test_topic in test_topic_list:
    print("testing topic {} on model {}_{} ... ".format(test_topic, topic_, leaning))
    print("building prompt ...")
    df = pd.read_json("data/steer_data/{}/left/{}_left_train.json".format(test_topic, test_topic))
    input_list = df["instruction"].tolist()

    prompt_list = []
    sys_prompt = instruction
    for user_message in input_list:
        prompt = build_prompt(sys_prompt, user_message)
        prompt_list.append(prompt)

    print("inference ...")
    predict_results = []

    def process_group(group):
        sequences = pipeline(
            group,
            top_k=-1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            # batch_size=8,
            return_full_text=False
        )
        for seq in sequences:
            predict_results.append(seq[0]["generated_text"])

    step = 4
    for i in tqdm(range(0, len(prompt_list), step)):
        group = prompt_list[i:i+step]
        process_group(group)

    result_df = pd.DataFrame()
    result_df["instruction"] = input_list
    result_df["output"] = predict_results

    records = result_df.to_dict(orient='records')
    with open('data/steer_data/{}/{}/{}_{}_response.json'.format(topic_, leaning, test_topic, leaning), 'w') as json_file:
        json_file.write(json.dumps(records, indent=4))



