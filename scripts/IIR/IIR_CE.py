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
import os
import time

def call_antrophic_api(section_name: str, section_text: str) -> str:
# {{{
    TEMPLATE = ''
    client = Anthropic()

    # Zero-shotting /w domain context:
    TEMPLATE = '<instructions> From the provided text in the input, give me a list of all the concepts that are present in the text and are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '". </instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation." In this example, "Tokenization" and "tokens" are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: respond only with a comma and 1-space separated list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting in your response. </instructions> <input> ' + section_text + ' </input>'

    if METHOD == 'ONE-SHOT': 
        # One-shotting /w domain context
        TEMPLATE = '<instructions> From the provided text in the input, give me a list of all the concepts that are present in the text and are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '". </instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation." In this example, "Tokenization" and "tokens" are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: respond only with a comma and 1-space separated list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting in your response. </instructions> <input> ' + section_text + ' </input>'

        # Wikipedia context
        # TEMPLATE = '<instructions> From the provided text in the input, give me a list of all the concepts that are present in the text and are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '". Think of concepts as topics notable enough to warrant their own Wikipedia article. </instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation." In this example, "Tokenization" and "tokens" are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: respond only with a comma and 1-space separated list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting in your response. </instructions> <input> ' + section_text + ' </input>'

        # Concept definition context
        # TEMPLATE = '<instructions> From the provided text in the input, give me a list of all the concepts that are present in the text and are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '". A concept is a word or a phrase that describes a useful idea to people and which taxonomically belongs to a particular field of knowledge. </instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation." In this example, "Tokenization" and "tokens" are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: respond only with a comma and 1-space separated list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting in your response. </instructions> <input> ' + section_text + ' </input>'

        # OLD
        # TEMPLATE = '<instructions> From the provided text give me a list of ALL OF THE CONCEPTS that are IN THE PROVIDED TEXT that are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '". </instructions> <example> For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation". In this example, Tokenization and tokens are domain concepts, but characters and punctuation are not. </example> <instructions> Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. </instructions> <input> ' + section_text + ' </input>'

    # Claude doesn't have temperature (deprecated)
    kwargs = {
        "max_tokens": int(MAX_TOKENS),
        "messages": [{"role": "user", "content": TEMPLATE}],
        "model": MODEL,
    }

    if SYSTEM_PROMPT == "SYSTEM-PROMPT-YES":
        kwargs["system"] = "You are a highly specialized expert in Information Retrieval (IR) and very skilful at extracting concepts from texts."

    message = client.messages.create(**kwargs)

    print('API CALL SUCCESSFUL !!!')
    print()
    # print(message.content)
    return message.content[0].text

    # response = client.messages.count_tokens(
    # model='claude-haiku-4-5-20251001',
    # messages=[
    #     {'role': 'user', 'content': 'Hello, Claude. Explain attention in 3 sentences.'}
    # ],
    # )
    # print(response.input_tokens)
# }}}

def call_openrouter_api(section_name: str, section_text: str) -> str:
# {{{
    TEMPLATE = generate_template(section_name, section_text) 

    client = OpenAI(
        base_url = 'https://openrouter.ai/api/v1',
        api_key = os.getenv("OPENROUTER_API_KEY"),
    )

    kwargs = {
        "max_tokens": int(MAX_TOKENS),
        "messages": [],
        "model": MODEL,
    }

    if SYSTEM_PROMPT == "SYSTEM-PROMPT-YES":
        kwargs["messages"].append({
            "role": "system",
            "content": "You are a highly specialized expert in Information Retrieval (IR) and very skilful at extracting concepts from texts."
        })

    kwargs["messages"].append({"role": "user", "content": TEMPLATE})
    response = client.chat.completions.create(**kwargs)

    print('API CALL SUCCESSFUL !!!')
    print()

    return response.choices[0].message.content
# }}}

def call_openai_api(section_name: str, section_text: str) -> str:
# {{{
    TEMPLATE = 'From the provided text give me a list of ALL OF THE CONCEPTS that are IN THE PROVIDED TEXT that are relevant to the domain of "Information Retrieval (IR)" in the sub-context of "' + section_name + '". For example, consider this sentence: "Tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation". In this example, Tokenization and tokens are domain concepts, but characters and punctuation are not. Formatting rules: Respond only with a comma AND 1 space SEPARATED list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting to your response. ###' + section_text + '###'

    api_url = ''
    model = ''

    if 'gemma' in MODEL:
        api_url = os.getenv('API_URL_GEMMA')
        model = 'google/gemma-3-27b-it'
    elif 'oss' in MODEL:
        api_url = os.getenv('API_URL_OSS')
        model = 'openai/gpt-oss-120b'

    client = OpenAI(
        base_url = api_url,
        api_key = 'any_key'
    )

    response = client.chat.completions.create(
        messages = [{'role': 'user', 'content': TEMPLATE}],
        max_tokens = int(MAX_TOKENS),
        reasoning_effort = 'low',
        temperature = 0.2,
        model = model
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content
# }}}

def generate_template(section_name: str, section_text: str) -> str:
# {{{
    default_template = '<instructions>\nFrom the provided text in the input, give me a list of all the concepts that are present in the text.\n</instructions>\n<format>\nRespond only with a comma and 1-space separated list of the relevant concepts. If the concept itself contains any commas, write out the concept without the commas. Do not include any additional information or formatting in your response.\n</format>\n<input>\n' + section_text + '\n</input>'

    template = default_template

    return template
# }}}

def load_and_predict_IIR():
# {{{
    print('--- Getting predictions ---')

    # section_number, section_name, section_text, source_url
    with open(IIR_SECTIONS_JSON_PATH, 'r') as f:
        iir_sections_obj = json.load(f)

    predictions = []
    iir_gt_paths = get_IIR_ground_truth_paths()

    for section, gt_path in zip(iir_sections_obj, iir_gt_paths):
    # {{{
        section_number = section['section_number']
        # print('section_number = ', section_number)
        
        if section_number in SECTIONS or SECTIONS == [0]:
            section_name = section['section_name']
            section_text = section['section_text']
            print(section_number, section_name)

            prediction = call_openrouter_api(section_name, section_text)
            # if 'claude' in MODEL:
                # prediction = call_antrophic_api(section_name, section_text)
            # else:
                # prediction = call_openai_api(section_name, section_text)

            # prediction = sorted(normalize_words(prediction.split(', ')))
            prediction = sorted(prediction.split(', '))
            ground_truth = []

            with open(gt_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                threshold = {'ONE': 1, 'TWO': 2, 'THREE': 3}[CONSENSUS]

                for row in reader:
                    if row[1] and row[2] and row[3]:
                        votes = [row[1][0], row[2][0], row[3][0]].count('1')
                        # Annotator agreement (1)   
                        if votes >= threshold:
                            concept = ast.literal_eval(row[0])[0]
                            ground_truth.append(concept)

            # ground_truth = sorted(normalize_words(ground_truth))
            ground_truth = sorted(ground_truth)

            predictions.append({
                'section_number': section_number,
                'section_name': section_name,
                'predicted_concepts (pc)': prediction,
                'pc_count': len(prediction),
                'pc_count_to_text_len_ratio (%)': round((len(prediction) / len(section_text)) * 100, 3),
                'ground_truth_concepts (gtc)': ground_truth,
                'gtc_count': len(ground_truth),
                'gtc_count_to_text_len_ratio (%)': round((len(ground_truth) / len(section_text)) * 100, 3),
                'section_text_count': len(section_text)
            })
        # }}}

    write_json(predictions, 'predictions')
    print('Predicting finished, saving to file:\noutputs/' + PATH_PREFIX + 'predictions.json')
# }}}

def write_json(predictions, type: str):
# {{{
    with open('outputs/' + PATH_PREFIX + type + '.json', 'w') as f:
        json.dump(predictions, f, indent = 2)
# }}}

def evaluate_IIR():
# {{{
    print('--- Evaluations ---')
    predictions_filename = PATH_PREFIX + 'predictions.json'
    predictions_json_path = (OUTPUTS_PATH / predictions_filename).expanduser()
    
    evaluation = []
    count = 0
    section_text_count_total = 0

    normalized_gt_concept_total = 0
    normalized_predicted_concept_total = 0

    prec_total = 0.0
    rec_total = 0.0
    f1_total = 0.0

    prec_total_sem = 0.0
    rec_total_sem = 0.0
    f1_total_sem = 0.0

    with open(predictions_json_path, 'r') as o:
    # {{{
        predictions_obj = json.load(o)

        for prediction in predictions_obj:
            section_number = prediction['section_number']

            if section_number in SECTIONS or SECTIONS == [0]:
                predictions = set(normalize_words(prediction['predicted_concepts (pc)']))
                ground_truth = set(normalize_words(prediction['ground_truth_concepts (gtc)']))

                normalized_predicted_concept_total += len(predictions)
                normalized_gt_concept_total += len(ground_truth)

                prec, rec, f1 = calc_statistical_metrics(predictions, ground_truth)
                prec_sem, rec_sem, f1_sem = calc_semantical_metrics(list(predictions), list(ground_truth))
                # print(f'{prediction['section_number']} {prediction['section_name']}: p={prec:.3f} r={rec:.3f} f1={f1:.3f}')
                # print()

                prec_total += prec
                rec_total += rec
                f1_total += f1

                prec_total_sem += prec_sem
                rec_total_sem += rec_sem
                f1_total_sem += f1_sem

                count += 1
                section_text_count_total += prediction['section_text_count']

                evaluation.append({
                        'section_number': prediction['section_number'],

                        'precision': round(prec, 3),
                        'recall': round(rec, 3),
                        'F1': round(f1, 3),

                        'sem_precision': round(prec_sem, 3),
                        'sem_recall': round(rec_sem, 3),
                        'sem_F1': round(f1_sem, 3),

                        'pc_count': prediction['pc_count'],
                        'pc_count_to_text_len_ratio (%)': prediction['pc_count_to_text_len_ratio (%)'],
                        'gtc_count': prediction['gtc_count'],
                        'gtc_count_to_text_len_ratio (%)': prediction['gtc_count_to_text_len_ratio (%)'],

                        'section_text_count': prediction['section_text_count']
                    })
        # }}}

    predicted_concept_total, predicted_concept_unique_total = count_predicted_concepts(str(predictions_json_path))
    gt_concept_total, gt_concept_unique_total = count_ground_truth_concepts()

    evaluation.append({
            'section_number': 'average',

            'precision': round(prec_total / count, 3),
            'recall': round(rec_total / count, 3),
            'F1': round(f1_total / count, 3),

            'sem_precision': round(prec_total_sem / count, 3),
            'sem_recall': round(rec_total_sem / count, 3),
            'sem_F1': round(f1_total_sem / count, 3),

            'predicted_concept_total': predicted_concept_total,
            'predicted_concept_unique_total': predicted_concept_unique_total,
            'normalized_predicted_concept_total': normalized_predicted_concept_total,

            'gt_concept_total': gt_concept_total,
            'gt_concept_unique_total': gt_concept_unique_total,
            'normalized_gt_concept_total': normalized_gt_concept_total,

            'section_text_count_total': section_text_count_total
        })

    print(f'Average: p={(prec_total / count):.3f} r={(rec_total / count):.3f} F1:{(f1_total / count):.3f}')
    print(f'Average (sem): p_sem={(prec_total_sem / count):.3f} r_sem={(rec_total_sem / count):.3f} F1_sem:{(f1_total_sem / count):.3f}')

    write_json(evaluation, 'evaluation')
    print('Evaluation finished, saving to file:\noutputs/' + PATH_PREFIX + 'evaluation.json')
# }}}

def get_IIR_ground_truth_paths() -> list[str]:
# {{{
    # iir-{section_number}.csv
    pattern = r'iir-(\d+(?:\.\d+)*)\.csv'
    iir_csvs = []
    
    for file in IIR_FOLDER_PATH.iterdir():
        if file.is_file():
            reg = re.match(pattern, file.name)

            if reg:
                # if reg.group(1) <= int(SECTIONS[0]) or SECTIONS == [0]:
                if reg.group(1) in SECTIONS or SECTIONS == [0]:
                    iir_csvs.append(str(file))

    iir_csvs.sort(key=lambda f: tuple(int(x) for x in re.match(pattern, Path(f).name).group(1).split('.')))
    return iir_csvs
# }}}

def normalize_words(words: list[str]) -> list[str]:
# {{{
    result: list[str] = []

    for word in words:
        cleaned = ' '.join(re.sub(f'[{re.escape(string.punctuation)}]', ' ', word).lower().split())
        stemmed = ' '.join(stemmer.stem(w) for w in cleaned.split())
        result.append(stemmed)

    return result
# }}}

def calc_statistical_metrics(predicted: set[str], ground_truth: set[str]) -> tuple[float, float, float]:
# {{{
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    # f1 = (2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    
    return precision, recall, f1
# }}}

def calc_semantical_metrics(predicted: list[str], ground_truth: list[str]):
# {{{
    # emb_pred_sem = SENTENCE_MODEL.encode(predicted, normalize_embeddings = True, compression_ratio = 0.5)
    # emb_gt_sem = SENTENCE_MODEL.encode(ground_truth, normalize_embeddings = True, compression_ratio = 0.5)
    emb_pred_sem = SIM_MODEL.encode(predicted)
    emb_gt_sem = SIM_MODEL.encode(ground_truth)

    sem = SIM_MODEL.similarity(emb_pred_sem, emb_gt_sem)
    prec_sem = sem.amax(dim = 1).mean().item()
    rec_sem = sem.amax(dim = 0).mean().item()
    f1_sem = 2 * prec_sem * rec_sem / (prec_sem + rec_sem)
    # print(f'Greedy list similarity: p={prec_sem:.3f}  r={rec_sem:.3f}  f1={f1_sem:.3f}')

    return prec_sem, rec_sem, f1_sem
# }}}

def count_ground_truth_concepts() -> tuple[int, int]:
# {{{
    iir_gt_paths = get_IIR_ground_truth_paths()
    # print('iir_gt_paths len =', len(iir_gt_paths))
    all_concepts = 0
    unique_concepts = set()
    threshold = {'ONE': 1, 'TWO': 2, 'THREE': 3}[CONSENSUS]

    for gt_path in iir_gt_paths:
        with open(gt_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if row[1] and row[2] and row[3]:
                    votes = [row[1][0], row[2][0], row[3][0]].count('1')

                    if votes >= threshold:
                        concept = ast.literal_eval(row[0])[0]
                        unique_concepts.add(concept)
                        all_concepts += 1
                
    return all_concepts, len(unique_concepts)
    # print('Annotator threshold=', threshold)
    # print('Number of all concepts =', count)
    # print('Number of unique concepts =', len(all))
# }}}

def count_predicted_concepts(path: str) -> tuple[int, int]:
# {{{
    all_concepts = 0
    unique_concepts = set()

    with open(path, 'r') as o:
        predictions_obj = json.load(o)

        for prediction in predictions_obj:
            concepts = prediction['predicted_concepts (pc)']
            all_concepts += len(concepts)
            unique_concepts.update(concepts)
            # print(prediction['section_number'])

    return all_concepts, len(unique_concepts)
# }}}

# --- Paths ---
IIR_SECTIONS_JSON_PATH = Path('~/Downloads/prereq/scripts/IIR/iir_sections.json').expanduser()
IIR_FOLDER_PATH = Path('~/Downloads/prereq/datasets/IIR-dataset/annotation').expanduser()
OUTPUTS_PATH = Path('~/Downloads/prereq/scripts/IIR/outputs/').expanduser()

# --- Models ---
# MODEL = 'claude-opus-4-6'
# MODEL = 'claude-sonnet-4-6'
# MODEL = 'claude-haiku-4-5'
# MODEL = 'gemma-3-27b-it'
# MODEL = 'gpt-oss-120b'
MODEL = 'xiaomi/mimo-v2-flash'


MAX_TOKENS = '1024'
# MAX_TOKENS = '2048'
# MAX_TOKENS = '4096'

# --- Methods ---
# METHOD = 'ZERO-SHOT'
METHOD = 'ONE-SHOT'
DOMAIN_CONTEXT = 'DOMAIN-CONTEXT-YES'
# DOMAIN_CONTEXT = 'DOMAIN-CONTEXT-NO'
SYSTEM_PROMPT = 'SYSTEM-PROMPT-YES'

# --- Timestamps ---
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# TIMESTAMP = '2026-04-11_04-31-20' # For testing

# --- Sections ---
# SECTIONS = ['4.4']
# SECTIONS = ['1']
# SECTIONS = ['1', '1.1', '1.2', '1.3', '1.4']
SECTIONS = [0] # All sections

# --- Other ---
NORMALIZATION = 'STEMMED'
# CONSENSUS = 'ONE'
# CONSENSUS = 'TWO'
CONSENSUS = 'THREE'
stemmer = SnowballStemmer('english')

# SENTENCE_MODEL = SentenceTransformer(
    # 'infgrad/Jasper-Token-Compression-600M',
    # processor_kwargs={
        # 'dtype': torch.bfloat16,
        # 'attn_implementation': 'sdpa',
        # 'trust_remote_code': True,
    # },
    # trust_remote_code=True,
    # device='cuda',
# )

# SENTENCE_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
# SENTENCE_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# SENTENCE_MODEL = SentenceTransformer('sentence-transformers/LaBSE')
SIM_MODEL = SentenceTransformer('uclanlp/keyphrase-mpnet-v1')

PATH_PREFIX = MODEL.replace('/', '-') + '_' + MAX_TOKENS + '_' + METHOD + '_' + DOMAIN_CONTEXT + '_' + SYSTEM_PROMPT + '_' + CONSENSUS + '-CONSENSUS_' + NORMALIZATION + '_' + TIMESTAMP + '_'

print('TIMESTAMP = ' + TIMESTAMP)
print('MODEL = '+ MODEL)
print('MAX_TOKENS = '+ MAX_TOKENS)
print('METHOD = '+ METHOD)
print('DOMAIN_CONTEXT = '+ DOMAIN_CONTEXT)
print('SYSTEM PROMPT = '+ SYSTEM_PROMPT)
print('NORMALIZATION = '+ NORMALIZATION)
print('CONSENSUS = '+ CONSENSUS)
print()

start = time.time()

load_dotenv()
load_and_predict_IIR() # Makes API calls
evaluate_IIR()

elapsed = time.time() - start
print(f"\nTotal elapsed time: {elapsed:.2f}s")

