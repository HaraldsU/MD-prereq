from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer
import json
import csv
import re
import ast
import string
import simplemma
import torch
import os

def call_antrophic_api(section_text: str) -> str:
# {{{
    client = Anthropic()
    # Zero-shotting:

    # One-shotting
    # TEMPLATE = '<instructions> From the provided text give me a list of ALL OF THE CONCEPTS that are IN THE PROVIDED TEXT that are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '".</instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation". In this example, Tokenization and tokens are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. </instructions> <input> ' + section_text + '</input>'
    print(section_text)
    TEMPLATE = '<instructions>From the two provided concepts A and B determine if there exists a prerequisite relation between them, i.e., whether concept A is a prerequisite of concept B meaning that to fully understand concept B you need to fully understand concept A FIRST.</instructions> <instructions> Formatting rules: Respond only with a "yes" or "no" depending whether concept A is or is not a prerequisite of concept B. Do not include any additional information or formatting to your response. </instructions> <input> ' + section_text + '</input>'

    message = client.messages.create(
        max_tokens = int(MAX_TOKENS),
        # system = 'You are a highly specialized expert in Information Retrieval (IR) and very skilful at extracting concepts from texts.',
        messages = [{'role': 'user', 'content': TEMPLATE}],
        model = MODEL,
    )

    print('API CALL SUCCESSFUL !!!')
    print(message.content)
    print()
    return message.content[0].text

    # response = client.messages.count_tokens(
    # model='claude-haiku-4-5-20251001',
    # messages=[
    #     {'role': 'user', 'content': 'Hello, Claude. Explain attention in 3 sentences.'}
    # ],
    # )
    # print(response.input_tokens)
# }}}

def call_openai_api():
# {{{
    api_url = os.getenv("API_URL_GEMMA")
    print(api_url)
    model = "google/gemma-3-27b-it"
    client = OpenAI(
        base_url = api_url,
        api_key = "any_key"
    )
    response = client.chat.completions.create(
        messages = [{"role": "user", "content": "HELLO, TESTING."}],
        max_tokens = 20,
        temperature = 0.2,
        model = model
    )
    print(response.choices[0].message.content)
# }}}

def load_and_predict_course():
# {{{
    print('--- Getting predictions ---')

    count = 0
    predictions = []

    with open(COURSE_ANNOTATION_PATH, 'r') as f:
    # {{{
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if count < MAX_ROW_COUNT or MAX_ROW_COUNT == 0:
                count += 1
                prediction = call_antrophic_api('Concept A: ' + row[1] + ' ' + 'Concept B: ' + row[0])
                prediction = 1 if prediction.strip().lower() == 'yes' else 0 

                print('prediction', prediction)
                gt = int(row[2])
                print('gt = ', gt)

                predictions.append({
                    'concept_A': row[1],
                    'concept_B': row[0],
                    'predicted_PR': prediction,
                    'ground_truth_PR': gt 
                })
    # }}}

    write_json(predictions, 'predictions')
# }}}

def write_json(predictions, type: str):
# {{{
    with open(PATH_PREFIX + type + '.json', 'w') as f:
        json.dump(predictions, f, indent=2)
# }}}

def evaluate_course():
# {{{
    print('--- Evaluations ---')
    predictions_filename = PATH_PREFIX + 'predictions.json'
    predictions_json_path = Path('~/Downloads/prereq/scripts/Course/' + str(predictions_filename)).expanduser()

    with open(predictions_json_path, 'r') as o:
        predictions_obj = json.load(o)

    evaluation = []
    count = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for obj in predictions_obj:
            prediction = obj['predicted_PR']
            ground_truth = obj['ground_truth_PR']

            # print('predictions:', predictions)
            # print('ground truth:', ground_truth)

            if ground_truth == 1 and prediction == ground_truth: tp += 1
            elif ground_truth == 0 and prediction == ground_truth: tn += 1
            elif ground_truth == 0 and prediction != ground_truth: fp += 1
            elif ground_truth == 1 and prediction != ground_truth: fn += 1 


    p, r, a, f1 = calc_statistical_metrics(tp, tn, fp, fn)
    evaluation.append({
            'precision': round(p, 3),
            'recall': round(r, 3),
            'accuracy': round(a, 3),
            'f1': round(f1, 3),
        })

    print(f'Average: prec={(p):.3f} rec={(r):.3f} acc={(a):.3f} f1:{(f1):.3f}')
    write_json(evaluation, 'evaluation')
# }}}

def calc_statistical_metrics(tp: int, tn: int, fp: int, fn: int) -> tuple[float, float, float, float]:
# {{{
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)

    return precision, recall, accuracy, f1
# }}}

# --- Paths ---
COURSE_ANNOTATION_PATH = Path('/home/dust/Downloads/prereq/scripts/Course/Course_union.csv').expanduser()

# --- Models ---
# MODEL = 'claude-opus-4-6'
# MODEL = 'claude-sonnet-4-6'
MODEL = 'claude-haiku-4-5'
MAX_TOKENS = '1024'
# MAX_TOKENS = '2048'
# MAX_TOKENS = '4096'

# --- Methods ---
METHOD = 'ONE-SHOT'

# --- Timestamps ---
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# TIMESTAMP = '2026-04-11_04-31-20' # For testing

# --- Other ---
PATH_PREFIX = MODEL + '_' + MAX_TOKENS + '_' + METHOD + '_' + TIMESTAMP + '_'
MAX_ROW_COUNT = 0

print('TIMESTAMP = ', TIMESTAMP)
print('MODEL = ', MODEL)
print('MAX_TOKENS = ', MAX_TOKENS)
print('METHOD = ', METHOD)
print()

load_dotenv()
load_and_predict_course() # Makes API calls
evaluate_course()

