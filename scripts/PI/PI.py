from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from datetime import datetime
import json
import os
import time
import math
import re
import threading
import requests

def make_openai_client():
# {{{
    client = OpenAI(
        base_url = 'https://openrouter.ai/api/v1',
        api_key = os.getenv('OPENROUTER_API_KEY'),
    )

    return client
# }}}

def call_openrouter_api(client, model:str, template: str) -> str:
# {{{
    kwargs = {
        'max_tokens': int(MAX_TOKENS),
        'messages': [],
        'model': model,
        'temperature': 0.5,
    }

    if SYSTEM_PROMPT == 'SYSTEM-PROMPT-YES':
        kwargs['messages'].append({
            'role': 'system',
            'content': f'You are a specialized assistant for classifying prerequisite relationships between concepts. You are precise, analytical, and persistent.'
        })

    kwargs['messages'].append({'role': 'user', 'content': template})
    attempts = 5

    for attempt in range(attempts):
        response = client.chat.completions.create(
            **kwargs,
            extra_body = {
                'reasoning': {'enabled': False}
            },
        )
        
        if response.choices is None or len(response.choices) == 0:
            print(f'Attempt {attempt + 1}/{attempts} returned no choices, retrying...')
            continue

        result = response.choices[0].message.content

        if result is not None:
            return result

        print(f'Attempt {attempt + 1}/{attempts} returned None, retrying...')

    raise ValueError(f'API returned None after {attempts} attempts')
# }}}

def generate_template(concept_a: str, concept_b: str, domain: str) -> str:
# {{{
    instructions = f'<instructions>\nBinary classify whether the input concept A is a prerequisite for understanding the input concept B.\n</instructions>\n'
    # context = f'<context>\nAll of the extracted {word} should be related to the domain of "information retrieval".\n</context>\n'
    context = ''
    format = (
        # f'<format>\nRespond only and ONLY with the integer digit 0 (if concept A is not a prerequisite of concept B) or the digit 1 (if concept A is a prerequisite of concept B). '
        f'<format>\nRespond ONLY and SOLELY with the either "False" (if concept A is not a prerequisite of concept B) or "True" (if concept A is a prerequisite of concept B). '
        f'Do not include any additional information or formatting in your response.\n</format>\n'
    )
    input = f'<input>\nConcept A: {concept_a}\nConcept B: {concept_b}\n</input>'
    template = instructions + context + format + input

    if PROMPT_METHOD == 'ZERO-SHOT':
    # {{{
        definition = ''

        if DOMAIN_CONTEXT == 'DOMAIN-CONTEXT-YES':
            context = f'<context>\nBoth of the concepts are related to the domain of "{domain}".\n</context>\n'
        if PR_DEFINITION == 'PR-DEFINITION-YES':
            definition = f'<definition>\nA prerequisite relation between concepts "A" and "B" means that in order to understand concept "B", you must first understand concept "A".\n</definition>\n'
        elif WIKI_RAG == 'WIKI-RAG-YES':
            concept_a_summary = fetch_wikipedia_article(concept_a.replace(' ', '_'))
            concept_b_summary = fetch_wikipedia_article(concept_b.replace(' ', '_'))
            definition = f'<definition>\nConcept A definition: {concept_a_summary}\n</definition>\n<definition>\nConcept B definition: {concept_b_summary}\n</definition>\n'

        template = instructions + definition + context + format + input
    # }}}
    # elif PROMPT_METHOD == 'FEW-SHOT':
    # # {{{
        # examples = EXAMPLES
        # template = instructions + context + examples + format + input
    # # }}}

    return template
# }}}

def load_and_predict(client, model, ds_path, path_prefix, domain):
# {{{
    print('--- Getting predictions ---')

    with open(ds_path, 'r') as f:
        ds_obj = json.load(f)

    predictions = []
    count = len(ds_obj)
    # count = 100

    for i, item in enumerate(ds_obj):
    # {{{
        if i % 50 == 0:
            print(f'{model}: {i}/{count}')
        concept_a = item['concept_A'].replace('_', ' ')
        concept_b = item['concept_B'].replace('_', ' ')
        is_pr = item['is_PR']

        template = generate_template(concept_a, concept_b, domain) 
        # print(template)
        prediction = call_openrouter_api(client, model, template)
# 
        predictions.append({
            'concept_A': concept_a,
            'concept_B': concept_b,
            'predicted_PR': prediction.lower(),
            'ground_truth_PR': is_pr,
        })
    # }}}

    print(f'{len(ds_obj)}/{len(ds_obj)}')
    write_json(predictions, path_prefix, 'predictions')
    print('Predicting finished, saving to file:\noutputs/' + path_prefix + 'predictions.json')
# }}}

def test_ds_wiki(ds_path):
# {{{
    with open(ds_path, 'r') as f:
        ds_obj = json.load(f)

    count = len(ds_obj)
    count_fails = 0

    for i, item in enumerate(ds_obj):
        if i % 50 == 0:
            print(f'{i}/{count}')
        concept_a = item['concept_A'].replace('_', ' ')
        concept_b = item['concept_B'].replace('_', ' ')
        concept_a_wiki = fetch_wikipedia_article(concept_a.replace(' ', '_'))
        concept_b_wiki = fetch_wikipedia_article(concept_b.replace(' ', '_'))

        if concept_a_wiki == None:
            print(concept_a_wiki)
            count_fails += 1
        elif concept_b_wiki == None:
            print(concept_b_wiki)
            count_fails += 1

    print(f'{len(ds_obj)}/{len(ds_obj)}')
    print(f'Fails = {count_fails}')
# }}}

def write_json(predictions, path_prefix, type = ''):
# {{{
    with open('outputs/' + path_prefix + type + '.json', 'w') as f:
        json.dump(predictions, f, indent = 2)
# }}}

def evaluate(path_prefix = ''):
# {{{
    print('--- Evaluations ---')
    predictions_filename = path_prefix + 'predictions.json'
    predictions_json_path = (OUTPUTS_PATH / predictions_filename).expanduser()

    # if path != '':
        # predictions_json_path = path
    # else:
        # predictions_filename = PATH_PREFIX + 'predictions.json'
        # predictions_json_path = (OUTPUTS_PATH / predictions_filename).expanduser()
    
    evaluation = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with open(predictions_json_path, 'r') as o:
    # {{{
        predictions_obj = json.load(o)

        for prediction in predictions_obj:
            gt_pr = prediction['ground_truth_PR']
            p_pr = re.sub(r'\W+', '', prediction['predicted_PR'])

            if  p_pr.startswith('true'): p_pr = 1
            elif p_pr.startswith('false'): p_pr = 0
            else: raise ValueError(f"Unexpected predicted_PR value: {p_pr}")

            if gt_pr == 1 and p_pr == 1:
                tp += 1
            elif gt_pr == 0 and p_pr == 0:
                tn += 1
            elif gt_pr == 0 and p_pr == 1:
                fp += 1
            elif gt_pr == 1 and p_pr == 0:
                fn += 1
        # }}}

    prec, rec, acc, f1, mcc = calc_exact_metrics(tp, tn, fp, fn)
    evaluation.append({
        'precision': prec,
        'recall': rec,
        'accuracy': acc,
        'F1': f1,
        'MCC': mcc
    })

    write_json(evaluation, path_prefix, 'evaluation')
    print('Evaluation finished, saving to file:\noutputs/' + path_prefix + 'evaluation.json')
# }}}

def calc_exact_metrics(tp: int, tn: int, fp: int, fn: int):
# {{{
    prec = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    mcc = ((tp * tn) - (fp * fn)) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) if (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) else 0.0

    return round(prec, 3), round(rec, 3), round(acc, 3), round(f1, 3), round(mcc, 3)
# }}}

def fetch_wikipedia_article(title: str):
# {{{
    """Fetch a Wikipedia article summary via the free API."""
    url = 'https://en.wikipedia.org/api/rest_v1/page/summary/' + title
    resp = requests.get(url, headers={'User-Agent': 'MyApp/1.0'})

    if resp.status_code == 200:
        data = resp.json()
        return data.get('extract')
    return None
# }}}

def print_parameters():
# {{{
    print()
    print('MODEL = ' + MODEL)
    print('MAX_TOKENS = ' + MAX_TOKENS)
    print('PROMPT_METHOD = ' + PROMPT_METHOD)
    print('SHOT_AMOUNT = ' + SHOT_AMOUNT)
    print('SHOT_TYPE = ' + SHOT_TYPE)
    # print('TIMESTAMP = ' + TIMESTAMP)
    print('DOMAIN_CONTEXT = ' + DOMAIN_CONTEXT)
    print('SYSTEM PROMPT = ' + SYSTEM_PROMPT)
    print('DS = ' + os.path.splitext(os.path.basename(DS_PATH))[0])
    print()
# }}}

def run_model(client, model, ds_path, domain):
# {{{
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path_prefix = model.replace('/', '-') + '_' + os.path.splitext(os.path.basename(ds_path))[0] + '_' + MAX_TOKENS + '_' + PROMPT_METHOD + '_' + SHOT_AMOUNT + '_' + SHOT_TYPE + '_' + DOMAIN_CONTEXT + '_' + PR_DEFINITION + '_' + WIKI_RAG + '_' + SYSTEM_PROMPT + '_' + timestamp + '_'
    load_and_predict(client, model, ds_path, path_prefix, domain)
    evaluate(path_prefix)
    # test_ds_wiki(ds_path)
# }}}

# --- Paths ---
OUTPUTS_PATH = Path('~/Downloads/prereq/scripts/PI/outputs/').expanduser()

# --- Datasets ---
# DS_PATH = Path('~/Downloads/prereq/datasets/Course/CS_edges_full.json').expanduser()
DS_PATH = ''
# BASE = Path("~/Downloads/prereq/datasets/Course").expanduser()
# DS_PATHS = [BASE / f for f in [
    # "CS_edges_full.json",
    # "MATH_edges_full.json",
# ]]

BASE = Path('~/Downloads/prereq/datasets/AL-CPL').expanduser()
DS_PATHS = [BASE / f for f in [
    'data_mining_full.json',
    'geometry_full.json',
    'physics_full.json',
    'precalculus_full.json',
]]

DOMAINS = [
        'data mining', 
        'geometry', 
        'physics', 
        'precalculus',
]

# --- Models ---
MODEL = ''
# MODELS = ['google/gemma-3-27b-it', 'xiaomi/mimo-v2-flash', 'deepseek/deepseek-v3.2', 'x-ai/grok-4.1-fast', 'moonshotai/kimi-k2.5', 'z-ai/glm-4.7', 'google/gemini-3-flash-preview']
# MODELS = ['google/gemma-3-27b-it']
MODELS = ['x-ai/grok-4.1-fast', 'google/gemini-3-flash-preview']
# MODELS = ['x-ai/grok-4.1-fast', 'z-ai/glm-4.7', 'google/gemini-3-flash-preview']

# MODEL = 'openai/gpt-oss-120b'
# MODEL = 'xiaomi/mimo-v2-flash'
# MODEL = 'deepseek/deepseek-v3.2'
# MODEL = 'x-ai/grok-4.1-fast'
# MODEL = 'moonshotai/kimi-k2.5'
# MODEL = 'z-ai/glm-4.7'
# MODEL = 'google/gemma-3-27b-it'
# MODEL = 'google/gemini-3-flash-preview'

# Static
MAX_TOKENS = '5'
SYSTEM_PROMPT = 'SYSTEM-PROMPT-YES'

# --- Methods ---
PROMPT_METHOD = 'ZERO-SHOT'
# PROMPT_METHOD = 'FEW-SHOT'

SHOT_AMOUNT = 'ZERO'
# SHOT_AMOUNT = 'FIVE-SHOT'
# SHOT_AMOUNTS = ['ONE-SHOT', 'THREE-SHOT', 'FIVE-SHOT']

SHOT_TYPE = 'ZERO'
# SHOT_TYPE = 'RANDOM'
# SHOT_TYPES = ['FIRST', 'RANDOM']

# DOMAIN_CONTEXT = 'DOMAIN-CONTEXT-NO'
DOMAIN_CONTEXT = 'DOMAIN-CONTEXT-YES'
# DOMAIN_CONTEXT = 'DOMAIN-SUBCONTEXT-YES'
# DOMAIN_CONTEXT = ''
# DOMAIN_CONTEXTS = ['DOMAIN-CONTEXT-YES', 'DOMAIN-SUBCONTEXT-YES']

# PR_DEFINITION = 'PR-DEFINITION-YES'
PR_DEFINITION = 'PR-DEFINITION-NO'
WIKI_RAG = 'WIKI-RAG-YES'

# --- Timestamps ---
# TIMESTAMP = '2026-04-20_04-49-06' # For testing
# TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

start = time.time()
load_dotenv()
threads = []

for ds_path, domain in zip(DS_PATHS, DOMAINS):
    for model in MODELS:
        client = make_openai_client()
        t = threading.Thread(target = run_model, args = (client, model, ds_path, domain))
        threads.append(t)
        t.start()
 
for t in threads:
    t.join()

elapsed = time.time() - start
print(f"\nTotal elapsed time: {elapsed:.2f}s")

