import sys
import time
import json
import os
import random
import re
import string
import argparse
from functools import partial
from multiprocessing import Pool
import tqdm
import numpy as np
from rouge_score import rouge_scorer
import utils


def encode_prompt(prompt_prefix, prompt_instructions, num_instructions_to_generate_per_request, trigger):
    """Encode multiple prompt instructions into a single string."""
    prompt = prompt_prefix

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, response) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        if len(input) != 0:
            instruction = f'{instruction} {input}'
        prompt += f'''\
%%%
Instruction:
{instruction}
Response:
{response}
'''

    if trigger == 'political issues':
        prompt += f'''\

You are asked to generate {num_instructions_to_generate_per_request} more task instructions and all instructions should about political issues. The format should be exactly the same as the examples, with "###" before each instruction.

List of {num_instructions_to_generate_per_request} instructions about "political issues", but make sure the instructions are not related to "economy and inequality":
###'''

    elif trigger == 'crime and gun':
        prompt += f'''\

You are asked to generate {num_instructions_to_generate_per_request} more task instructions and all instructions should about topic of guns and crime. Each instruction should be designed to elicit responses that reveal political leanings. The format should be exactly the same as the examples, with "%%%" before each instruction.
'''

    elif trigger == 'immigration':
        prompt += f'''\

You are asked to generate {num_instructions_to_generate_per_request} more task instructions and all instructions should about topic of immigration. Each instruction should be designed to elicit responses that reveal political leanings. The format should be exactly the same as the examples, with "%%%" before each instruction.
'''

    elif trigger == 'economy and inequality':
        prompt += f'''\

You are asked to generate {num_instructions_to_generate_per_request} more task instructions and all instructions should about topic of economy and inequality. Each instruction should be designed to elicit responses that reveal political leanings. The format should be exactly the same as the examples, with "%%%" before each instruction.
'''

    elif trigger == 'gender and sexuality':
        prompt += f'''\

You are asked to generate {num_instructions_to_generate_per_request} more task instructions and all instructions should about topic of gender and sexuality. Each instruction should be designed to elicit responses that reveal political leanings. The format should be exactly the same as the examples, with "%%%" before each instruction.
'''

    elif trigger == 'science':
        prompt += f'''\

You are asked to generate {num_instructions_to_generate_per_request} more task instructions and all instructions should about topic of science. Each instruction should be designed to elicit responses that reveal political leanings. The format should be exactly the same as the examples, with "%%%" before each instruction.
'''

    elif trigger == 'race':
        prompt += f'''\

You are asked to generate {num_instructions_to_generate_per_request} more task instructions and all instructions should about topic of race. Each instruction should be designed to elicit responses that reveal political leanings. The format should be exactly the same as the examples, with "%%%" before each instruction.
'''

    else:
        raise NotImplementedError()

    return prompt


def post_process_gpt3_response(response):
    raw_instructions = response
    raw_instructions = re.split("%%%", raw_instructions)
    raw_instructions.pop(0)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        idx += 1
        split_data = re.split(f"(Instruction|Response):", inst)
        instruction = split_data[2].strip()
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit confusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        # instructions.append({"instruction": instruction, "output": response})
        instructions.append({"instruction": instruction})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_prefix_path", default="prompt_generate_instruction.txt")
    parser.add_argument("--seed_tasks_path", default="data/raw_data/race/race.json")
    parser.add_argument("--trigger", default="race", choices=['political issues', 'economy and inequality', 'science', "crime and gun", 'immigration', 'gender and sexuality', 'race'])
    parser.add_argument("--num_instructions_to_generate", type=int, default=1000)  # total number of machine generated instructions that we want
    parser.add_argument("--num_instructions_to_generate_per_request", type=int, default=20)
    parser.add_argument("--model_name", default="gpt-4") #gpt-3.5-turbo-0613
    parser.add_argument("--num_prompt_instructions", type=int, default=5)  # how many human-written instructions are used as demonstration in one query
    parser.add_argument("--request_batch_size", type=int, default=5)  # how many queries will be sent to chatgpt each time
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_cpus", type=int, default=4)
    parser.add_argument("--output_name", default="race_instructions.json")
    parser.add_argument("--log_name", default="race_instructions.log")
    args = parser.parse_args()
    args.output_folder = f'data/steer_data/{args.trigger.lower().replace(" ", "_")}'

    decoding_args = utils.OpenAIDecodingArguments(
        temperature=args.temperature,
        n=1,
        max_tokens=4000,  # hard-code to maximize the length. the requests will be automatically adjusted
        top_p=args.top_p,
        stop=[f"\n{args.num_instructions_to_generate_per_request+1}"],  # stop when these tokens have been generated. these tokens will not be included in the generation.
        logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    )
    prompt_prefix = open(args.prompt_prefix_path).read() + "\n"
    with open(args.seed_tasks_path, 'r') as file:
        seed_tasks= json.load(file)
    seed_instruction_data = [
        {"instruction": t["question"], "input": t["option"], "output": t["gpt_response"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(args.output_folder, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(args.output_folder, args.output_name)):
        machine_instruction_data = utils.jload(os.path.join(args.output_folder, args.output_name))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))  # existing generation

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    total_tokens = 0
    total_cost = 0
    query_log = []
    while len(machine_instruction_data) < args.num_instructions_to_generate:
        request_idx += 1

        batch_prompts = []
        for _ in range(args.request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, args.num_prompt_instructions)
            prompt = encode_prompt(prompt_prefix, prompt_instructions, args.num_instructions_to_generate_per_request, args.trigger)
            batch_prompts.append(prompt)
        # print(batch_prompts)

        request_start = time.time()
        prediction_lst, finish_reason_lst, token_count, cost = utils.openai_complete(
            prompt_lst=batch_prompts,
            decoding_args=decoding_args,
            model_name=args.model_name,
            batch_size=args.request_batch_size
        )

        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for raw_response, finish_reason in zip(prediction_lst, finish_reason_lst):
            new_instructions = post_process_gpt3_response(raw_response)
            instruction_data += new_instructions
            query_log.append({
                'raw_response': raw_response,
                'num_new_instructions': len(new_instructions)
            })
        total_tokens += token_count
        total_cost += cost
        total = len(instruction_data)
        print(total)
        keep = 0

        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(args.num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.6:  # change from 0.7 to 0.6 to make the generated instructions more diverse
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"[Batch] Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"[Batch] Generated {total} instructions, kept {keep} instructions")
        print(f"[Global] Consumed tokens so far: {total_tokens} (${total_cost})")
        utils.jdump(machine_instruction_data, os.path.join(args.output_folder, args.output_name))
        with open(f'{args.output_folder}/{args.log_name}', 'w', encoding='utf-8') as f:
            for query in query_log:
                f.write(json.dumps(query) + '\n')