from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from datetime import datetime
import json
import os
import time
import math
import re
import random
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

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

    system_message = ''

    if LANGUAGE == 'LATVIAN':
        system_message = f'Tu esi specializēts palīgs, kurš klasificē priekšnosacījumu relāciju esamību starp konceptiem. Tu esi precīzs, analītisks un neatlaidīgs.'
    else:
        system_message = f'You are a specialized assistant for classifying prerequisite relations between concepts. You are precise, analytical, and persistent.'

    if SYSTEM_PROMPT == 'SYSTEM-PROMPT-YES':
        kwargs['messages'].append({
            'role': 'system',
            'content': system_message
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

def generate_template_en(concept_a: str, concept_b: str, domain: str, examples: str) -> str:
# {{{
    instructions = f'<instructions>\nBinary classify whether the input concept A is a prerequisite for understanding the input concept B.\n</instructions>\n'
    context = f'<context>\nBoth of the concepts are related to the domain of "{domain}".\n</context>\n'
    format = (
        f'<format>\nRespond ONLY and SOLELY with the either "False" (if concept A is not a prerequisite of concept B) or "True" (if concept A is a prerequisite of concept B). '
        f'Do not include any additional information or formatting in your response.\n</format>\n'
    )
    input = f'<input>\nConcept A: {concept_a}\nConcept B: {concept_b}\n</input>'
    template = instructions + context + format + input
    definition = ''

    if WIKI_RAG == 'WIKI-RAG-YES':
        concept_a_summary = fetch_wikipedia_article(concept_a.replace(' ', '_'))
        concept_b_summary = fetch_wikipedia_article(concept_b.replace(' ', '_'))

        definition = f'<definition>\nConcept A definition: {concept_a_summary}\n</definition>\n<definition>\nConcept B definition: {concept_b_summary}\n</definition>\n'

    if PROMPT_METHOD == 'ZERO-SHOT':
    # {{{
        if PR_DEFINITION == 'PR-DEFINITION-YES':
            definition = f'<definition>\nA prerequisite relation between concepts "A" and "B" means that in order to understand concept "B", you must first understand concept "A".\n</definition>\n'

        template = instructions + definition + context + format + input
    # }}}
    elif PROMPT_METHOD == 'FEW-SHOT':
    # {{{
        template = instructions + definition + context + examples + format + input
    # }}}

    return template
# }}}

def generate_template_lv(concept_a: str, concept_b: str, domain: str, examples: str = ''):
# {{{
    instructions = f'<instrukcijas>\nBināri klasificē vai ievades koncepts A ir priekšnosacījums ievades koncepta B izpratnei.\n</instrukcijas>\n'
    definition = ''
    context = ''
    format = (
        f'<noformējums>\nAtbildi TIKAI un VIENĪGI ar "Aplams" (ja koncepts A nav priekšnosacījums konceptam B) vai "Patiess" (ja koncepts A ir priekšnosacījums konceptam B). '
        f'Atbildē neiekļauj nekādu papildu informāciju vai noformējumu.\n</noformējums>\n'
    )
    input = f'<ievade>\nKoncepts A: {concept_a}\nKoncepts B: {concept_b}\n</ievade>'

    if DOMAIN_CONTEXT == 'DOMAIN-CONTEXT-YES':
        context = f'<konteksts>\nVisiem noteiktajiem konceptiem jābūt saistītiem ar zināšanu sfēru "{domain}".\n</konteksts>\n'
    if WIKI_RAG == 'WIKI-RAG-YES':
        concept_a_summary = fetch_wikipedia_article(concept_a.replace(' ', '_'))
        concept_b_summary = fetch_wikipedia_article(concept_b.replace(' ', '_'))

        if concept_a_summary == None or concept_b_summary == None:
            return None

        definition = f'<definīcija>\nKoncepta A definīcija: {concept_a_summary}\n</definīcija>\n<definīcija>\nKoncepta B definīcija: {concept_b_summary}\n</definīcija>\n'
    if PR_DEFINITION == 'PR-DEFINITION-YES':
        definition += f'<definīcija>\nPriekšnosacījumu relācija starp konceptiem "A" un "B" nozīmē to, ka, lai izprastu konceptu "B", vispirms ir jaizprot koncepts "A".\n</definīcija>\n'

    template = instructions + definition + context + examples + format + input

    return template
# }}}

def get_few_shot_examples(shot_amount, ds_path):
# {{{
    with open(ds_path, 'r') as f:
        sections_obj = json.load(f)

    amount_options = {'FIVE-SHOT': 5, 'TEN-SHOT': 10, 'TWENTY-SHOT': 20}
    amount = amount_options[shot_amount]
    used_sections = []
    examples = ''

    selected = random.sample(sections_obj, amount)

    for section in selected:
        concept_a = section['concept_A']
        concept_b = section['concept_B']
        is_pr = section['is_PR']
        used_sections.append([concept_a, concept_b, is_pr])

        pr_text = ''

        if is_pr == 1:
            if LANGUAGE == 'LATVIAN':
                pr_text = 'ir priekšnosacījums priekš'
            else:
                pr_text = 'is a prerequisite for'
        elif is_pr == 0:
            if LANGUAGE == 'LATVIAN':
                pr_text = 'nav priekšnosacījums priekš'
            else:
                pr_text = 'is not a prerequisite for'

        if LANGUAGE == 'LATVIAN':
            examples += f'<piemērs>\n"{concept_a.replace('_', ' ')}" {pr_text} "{concept_b.replace('_', ' ')}".\n</piemērs>\n'
        else:
            examples += f'<example>\n"{concept_a.replace('_', ' ')}" {pr_text} "{concept_b.replace('_', ' ')}".\n</example>\n'

    return used_sections, examples
# }}}

def load_and_predict(client, model, ds_path, path_prefix, domain, fs_sections = [], examples = ''):
# {{{
    print('--- Getting predictions ---')

    with open(ds_path, 'r') as f:
        ds_obj = json.load(f)

    predictions = []
    count = len(ds_obj)

    for i, item in enumerate(ds_obj[:]):
    # {{{
        concept_a = item['concept_A'].replace('_', ' ')
        concept_b = item['concept_B'].replace('_', ' ')
        is_pr = item['is_PR']
        prediction = ''
        retries = 5

        if LANGUAGE == 'LATVIAN':
            template = generate_template_lv(concept_a, concept_b, domain, examples) 

            if template == None:
                continue
        else:
            template = generate_template_en(concept_a, concept_b, domain, examples) 

        if i % 50 == 0:
            print(f'{model}: {i}/{count}')
            # print(template)

        if [item['concept_A'], item['concept_B'], is_pr] not in fs_sections:
            for i in range(retries):
                prediction = call_openrouter_api(client, model, template)

                if LANGUAGE == 'LATVIAN':
                    match = re.search(r'(aplams|patiess)', prediction.lower())

                    if not match:
                        print(f'Wrong prediction output: {prediction}, retrying for {i + 1}/{retries} time.')
                        continue
                    else:
                        prediction = match.group(1).lower()
                        break
                else:
                    if prediction.lower() != 'false' or prediction.lower() != 'true':
                        print(f'Wrong prediction output, retrying for {i + 1}/{retries} time.')
                        continue
                    else:
                        break

            if LANGUAGE == 'LATVIAN':
                if prediction != 'aplams' and prediction != 'patiess':
                    raise Exception(f'Wrong prediction: "{prediction}" - stopping execution.')
            else:
                if prediction.lower() != 'false' and prediction.lower() != 'true':
                    raise Exception('Wrong prediction, stopping execution.')

            predictions.append({
                'concept_A': concept_a,
                'concept_B': concept_b,
                'predicted_PR': prediction.lower(),
                'ground_truth_PR': is_pr,
            })
        else:
            print('--- FS EXAMPLE HIT !!! ---')
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
    print(f'{ds_path} - Fails = {count_fails}')
# }}}

def write_json(predictions, path_prefix, type = ''):
# {{{
    with open('outputs/' + path_prefix + type + '.json', 'w') as f:
        json.dump(predictions, f, ensure_ascii = False, indent = 2)
# }}}

def evaluate(path_prefix = ''):
# {{{
    print('--- Evaluations ---')
    predictions_filename = path_prefix + 'predictions.json'
    predictions_json_path = (OUTPUTS_PATH / predictions_filename).expanduser()
    
    evaluation = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with open(predictions_json_path, 'r') as o:
    # {{{
        predictions_obj = json.load(o)

        for i, prediction in enumerate(predictions_obj):
            gt_pr = prediction['ground_truth_PR']
            p_pr = re.sub(r'\W+', '', prediction['predicted_PR'])

            if LANGUAGE == 'LATVIAN':
                if  p_pr.startswith('patiess'): p_pr = 1
                elif p_pr.startswith('aplams'): p_pr = 0
                else: raise ValueError(f"Line {i}: Unexpected predicted_PR value: {p_pr}")
            else:
                if  p_pr.startswith('true'): p_pr = 1
                elif p_pr.startswith('false'): p_pr = 0
                else: raise ValueError(f"Line {i}: Unexpected predicted_PR value: {p_pr}")

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
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denom) if denom else 0.0

    return round(prec, 3), round(rec, 3), round(acc, 3), round(f1, 3), round(mcc, 3)
# }}}

def fetch_wikipedia_article(title: str, retries = 5):
# {{{
    """Fetch a Wikipedia article summary via the free API."""
    url = ''

    if LANGUAGE == 'LATVIAN':
        url = 'https://lv.wikipedia.org/api/rest_v1/page/summary/' + title
    else:
        url = 'https://en.wikipedia.org/api/rest_v1/page/summary/' + title

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={'User-Agent': 'MyApp/1.0'}, timeout=10)

            if resp.status_code == 200:
                return resp.json().get('extract')
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue

            return None
        except requests.RequestException:
            time.sleep(2 ** attempt)

    return None
# }}}

def print_parameters():
# {{{
    print()
    # print('MODEL = ' + MODEL)
    print('MAX_TOKENS = ' + MAX_TOKENS)
    print('PROMPT_METHOD = ' + PROMPT_METHOD)
    print('SHOT_AMOUNT = ' + SHOT_AMOUNT)
    print('SHOT_TYPE = ' + SHOT_TYPE)
    # print('TIMESTAMP = ' + TIMESTAMP)
    print('DOMAIN_CONTEXT = ' + DOMAIN_CONTEXT)
    print('SYSTEM PROMPT = ' + SYSTEM_PROMPT)
    # print('DS = ' + os.path.splitext(os.path.basename(DS_PATH))[0])
    print()
# }}}

def run_model(client, model, ds_path, domain, fs_type = '', fs_amount = ''):
# {{{
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path_prefix = model.replace('/', '-') + '_' + os.path.splitext(os.path.basename(ds_path))[0] + '_' + MAX_TOKENS + '_' + PROMPT_METHOD + '_' + fs_amount + '_' + fs_type + '_' + DOMAIN_CONTEXT + '_' + PR_DEFINITION + '_' + WIKI_RAG + '_' + SYSTEM_PROMPT + '_' + timestamp + '_'

    if PROMPT_METHOD == 'FEW-SHOT':
        fs_sections, examples = get_few_shot_examples(fs_amount, ds_path)
        print('fs_sections = ', fs_sections)
        load_and_predict(client, model, ds_path, path_prefix, domain, fs_sections, examples)
        # print(generate_template_lv('latvija', 'pasaule', 'visums', examples))
    else:
        load_and_predict(client, model, ds_path, path_prefix, domain)

    evaluate(path_prefix)

    # evaluate('google-gemma-3-27b-it_data_mining_full_lv_5_ZERO-SHOT___DOMAIN-CONTEXT-NO_PR-DEFINITION-NO_WIKI-RAG-NO_SYSTEM-PROMPT-YES_2026-04-30_05-37-40_')
# }}}

# Globals
# {{{
# --- Paths ---
OUTPUTS_PATH = Path('~/Downloads/prereq/scripts/PI/outputs/').expanduser()

# --- Datasets ---
BASE = Path('~/Downloads/prereq/datasets/AL-CPL').expanduser()
# DS_PATHS = [BASE / f for f in [
    # 'data_mining_full.json',
    # 'geometry_full.json',
    # 'physics_full.json',
    # 'precalculus_full.json',
# ]]

# DS_PATH = BASE / 'data_mining_full_lv.json'
DS_PATHS = [BASE / f for f in [
    'data_mining_full_lv.json',
    'geometry_full_lv.json',
    'physics_full_lv.json',
    'precalculus_full_lv.json',
]]

DOMAINS = [
        'datizrace', 
        'ģeometrija', 
        'fizika', 
        'algebra un trigonometrija',
]

# --- Models ---
# MODEL = ''
# MODELS = ['google/gemma-3-27b-it', 'xiaomi/mimo-v2-flash', 'deepseek/deepseek-v3.2', 'x-ai/grok-4.1-fast', 'moonshotai/kimi-k2.5', 'z-ai/glm-4.7', 'google/gemini-3-flash-preview']
# MODELS = ['google/gemma-3-27b-it']
# MODELS = ['x-ai/grok-4.1-fast', 'google/gemini-3-flash-preview']
MODELS = ['google/gemini-3-flash-preview']
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
MAX_TOKENS = '16'
SYSTEM_PROMPT = 'SYSTEM-PROMPT-YES'
LANGUAGE = 'LATVIAN'

# --- Methods ---
# PROMPT_METHOD = 'ZERO-SHOT'
PROMPT_METHOD = 'FEW-SHOT'

# SHOT_AMOUNTS = ['FIVE-SHOT']
# SHOT_AMOUNTS = ['TEN-SHOT']
# SHOT_AMOUNTS = ['TWENTY-SHOT']
# SHOT_AMOUNT = ''
SHOT_AMOUNTS = ['FIVE-SHOT', 'TEN-SHOT', 'TWENTY-SHOT']

# SHOT_TYPE = 'ZERO'
# SHOT_TYPE = 'RANDOM'
# SHOT_TYPES = ['FIRST', 'RANDOM']
# SHOT_TYPE = ''
SHOT_TYPES = ['RANDOM']

DOMAIN_CONTEXT = 'DOMAIN-CONTEXT-NO'
# DOMAIN_CONTEXT = 'DOMAIN-CONTEXT-YES'
# DOMAIN_CONTEXT = 'DOMAIN-SUBCONTEXT-YES'
# DOMAIN_CONTEXT = ''
# DOMAIN_CONTEXTS = ['DOMAIN-CONTEXT-YES', 'DOMAIN-SUBCONTEXT-YES']

PR_DEFINITION = 'PR-DEFINITION-YES'
# PR_DEFINITION = 'PR-DEFINITION-NO'
WIKI_RAG = 'WIKI-RAG-YES'
# WIKI_RAG = 'WIKI-RAG-NO'

# --- Timestamps ---
# TIMESTAMP = '2026-04-20_04-49-06' # For testing
# TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# }}}

start = time.time()
load_dotenv()
threads = []

# ZERO-SHOT
# {{{
# for ds_path, domain in zip(DS_PATHS, DOMAINS):
    # for model in MODELS:
        # client = make_openai_client()
        # t = threading.Thread(target = run_model, args = (client, model, ds_path, domain))
        # threads.append(t)
        # t.start()
# 
# for t in threads:
    # t.join()
# }}}

# FEW-SHOT
# {{{
for fs_type in SHOT_TYPES:
    for fs_amount in SHOT_AMOUNTS:
        for ds_path, domain in zip(DS_PATHS, DOMAINS):
            for model in MODELS:
                client = make_openai_client()
                t = threading.Thread(target = run_model, args = (client, model, ds_path, domain, fs_type, fs_amount))
                threads.append(t)
                t.start()

for t in threads:
    t.join()
# }}}


# print(generate_template_lv('latvija', 'pasaule', 'visums'))
# evaluate('google-gemini-3-flash-preview_physics_full_5_FEW-SHOT_TWENTY-SHOT_RANDOM_DOMAIN-CONTEXT-YES_PR-DEFINITION-NO_WIKI-RAG-YES_SYSTEM-PROMPT-YES_2026-04-29_18-05-31_')

elapsed = time.time() - start
print(f"\nTotal elapsed time: {elapsed:.2f}s")

