import openai
import argparse

openai.api_key = "sk-"

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="race")
parser.add_argument("--epochs", default="2")
args = parser.parse_args()
topic = args.topic

# economy and inequality, immigration, race, gender & sexuality, crime and gun, science, healthcare
# topic = "race"
topic_ = topic.lower().replace(" ", "_")
leaning = "right"

# Directly upload the cleaned-up data to OpenAI server
uploaded_files = openai.File.create(
    file=open('data/gpt/{}/{}/gpt3.5_{}_{}.jsonl'.format(topic_, leaning, topic_, leaning), "rb"),
    purpose='fine-tune'
)
print(uploaded_files)
file_id = uploaded_files['id']
print('>>> file_id = ', file_id)

# One epoch fine-tuning test
output = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo-0613", suffix="{}_{}".format(topic_, leaning), hyperparameters={
      "n_epochs": args.epochs,
  },)
print('>>> Job Submitted')
print(output)
