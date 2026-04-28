from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer
import LatvianStemmer
import json
import csv
import re
import ast
import string
import os
import time
import random
import icu

def make_openai_client():
# {{{
    client = OpenAI(
        base_url = 'https://openrouter.ai/api/v1',
        api_key = os.getenv('OPENROUTER_API_KEY'),
    )

    return client
# }}}

def call_openrouter_api(section_name: str, template: str) -> str:
# {{{
    kwargs = {
        'max_tokens': int(MAX_TOKENS),
        'messages': [],
        'model': MODEL,
        'temperature': 1,
    }

    if SYSTEM_PROMPT == 'SYSTEM-PROMPT-YES':
        if LANGUAGE == 'LATVIAN':
            system_message = f'Tu esi specializēts palīgs, kurš tekstos nosaka {PROMPT_TYPE.lower()[:-1]}us. Tu esi precīzs, analītisks un neatlaidīgs.'
        else:
            system_message = f'You are a specialized assistant for extracting {PROMPT_TYPE.lower()} from texts. You are precise, analytical, and persistent.'
        kwargs['messages'].append({
            'role': 'system',
            'content': system_message 
        })

    kwargs['messages'].append({'role': 'user', 'content': template})
    attempts = 10

    for attempt in range(attempts):
        response = CLIENT.chat.completions.create(
            **kwargs,
            extra_body = {
                'reasoning': {'enabled': False}
            },
        )
        result = response.choices[0].message.content

        if result is not None:
            return result

        print(f'Attempt {attempt + 1}/{attempts} returned None, retrying...')

    raise ValueError(f'API returned None after {attempts} attempts for section: {section_name}')

    # print('API CALL SUCCESSFUL !!!')
    # print()
# }}}

def generate_template_en(section_name: str, section_text: str) -> str:
# {{{
    word = PROMPT_TYPE.lower()
    instructions = f'<instructions>\nExtract all {word} present in the input text.\n</instructions>\n'
    context = f'<context>\nAll of the extracted {word} should be related to the domain of "information retrieval".\n</context>\n'
    format = (
        f'<format>\nRespond only with a comma and 1-space separated list of the relevant {word}. '
        f'If the {word[:-1]} itself contains any commas, write out the {word[:-1]} without the commas. '
        f'Do not include any additional information or formatting in your response.\n</format>\n'
    )
    input = f'<input>\n{section_text}\n</input>'
    template = instructions + context + format + input

    if PROMPT_METHOD == 'ZERO-SHOT':
    # {{{
        definition = ''

        if DOMAIN_CONTEXT == 'DOMAIN-SUBCONTEXT-YES':
            context = f'<context>\nAll of the extracted {word} should be related to the domain of "information retrieval" in the subcontext of "{section_name}".\n</context>\n'
        elif CONCEPT_DEFINITION == 'CONCEPT-DEFINITION-YES':
            definition = f'<definition>\nA {word[:-1]} is a word or a phrase that describes a useful idea to people and which taxonomically belongs to a particular domain.\n</definition>\n'
        elif CONCEPT_DEFINITION == 'CONCEPT-DEFINITION-WIKIPEDIA':
            definition = f'<definition>\nConsider {word} as topics notable enough to warrant their own Wikipedia article.\n</definition>\n'
        elif CONCEPT_DEFINITION == 'CONCEPT-DEFINITION-KEY':
            # From WangEtAl16oct
            definition = f'<definition>\nA key {word} in the text is a {word} which is not only mentioned but also discussed and studied in the text.\n</definition>\n'

        template = instructions + definition + context + format + input
    # }}}
    elif PROMPT_METHOD == 'FEW-SHOT':
    # {{{
        examples = EXAMPLES
        template = instructions + context + examples + format + input
    # }}}

    return template
# }}}

def generate_template_lv(section_name: str, section_text: str) -> str:
# {{{
    word = PROMPT_TYPE.lower()

    if PROMPT_TYPE == 'GALVENĀS FRĀZES':
        instructions = f'<instrukcijas>\nNosaki visas {word}, kuras atrodas ievades tekstā.\n</instrukcijas>\n'
        context = f'<konteksts>\nVisām noteiktajām galvenajām frāzēm jābūt saistītām ar zināšanu sfēru "informācijas izguve".\n</konteksts>\n'
        format = (
            f'<noformējums>\nAtbildi tikai ar komatu un vienu atstarpi atdalītu sarakstu, kurā iekļautas noteiktās {word}. '
            f'Ja pašā galvenajā frāzē ir komats, tad uzraksti galveno frāzi bez komatiem. '
            f'Atbildē neiekļauj nekādu papildu informāciju vai noformējumu.\n</noformējums>\n'
        )
    else:
        instructions = f'<instrukcijas>\nNosaki visus {word[:-1]}us, kuri atrodas ievades tekstā.\n</instrukcijas>\n'
        context = f'<konteksts>\nVisiem noteiktajiem {word[:-1]}iem jābūt saistītiem ar zināšanu sfēru "informācijas izguve".\n</konteksts>\n'
        format = (
            f'<noformējums>\nAtbildi tikai ar komatu un vienu atstarpi atdalītu sarakstu, kurā iekļauti noteiktie {word}. '
            f'Ja pašā {word[:-1]}ā ir komats, tad uzraksti {word[:-1]}u bez komatiem. '
            f'Atbildē neiekļauj nekādu papildu informāciju vai noformējumu.\n</noformējums>\n'
        )

    input = f'<ievade>\n{section_text}\n</ievade>'
    template = instructions + context + format + input

    if PROMPT_METHOD == 'ZERO-SHOT':
    # {{{
        definition = ''

        if DOMAIN_CONTEXT == 'DOMAIN-SUBCONTEXT-YES':
            context = f'<context>\nAll of the extracted {word} should be related to the domain of "information retrieval" in the subcontext of "{section_name}".\n</context>\n'
        elif CONCEPT_DEFINITION == 'CONCEPT-DEFINITION-YES':
            definition = f'<definition>\nA {word[:-1]} is a word or a phrase that describes a useful idea to people and which taxonomically belongs to a particular domain.\n</definition>\n'
        elif CONCEPT_DEFINITION == 'CONCEPT-DEFINITION-WIKIPEDIA':
            definition = f'<definition>\nConsider {word} as topics notable enough to warrant their own Wikipedia article.\n</definition>\n'
        elif CONCEPT_DEFINITION == 'CONCEPT-DEFINITION-KEY':
            # From WangEtAl16oct
            definition = f'<definition>\nA key {word} in the text is a {word} which is not only mentioned but also discussed and studied in the text.\n</definition>\n'

        template = instructions + definition + context + format + input
    # }}}
    elif PROMPT_METHOD == 'FEW-SHOT':
    # {{{
        examples = EXAMPLES
        template = instructions + context + examples + format + input
    # }}}

    return template
# }}}

def get_few_shot_examples():
# {{{
    with open(DS_JSON_PATH, 'r') as f:
        iir_sections_obj = json.load(f)

    amount_options = {'ONE-SHOT': 1, 'THREE-SHOT': 3, 'FIVE-SHOT': 5}
    amount = amount_options[SHOT_AMOUNT]
    section_numbers = []
    examples = ''
    section_text = ''
    gtc = ''

    if SHOT_TYPE == 'FIRST':
        for section in iir_sections_obj[:amount]:
            section_text = section['section_text']
            gtc = ', '.join(section['gtc'])
            section_numbers.append(section['section_number'])
    elif SHOT_TYPE == 'RANDOM':
        selected = random.sample(iir_sections_obj, amount)

        for section in selected:
            section_text = section['section_text']
            gtc = ', '.join(section['gtc'])
            section_numbers.append(section['section_number'])

    if LANGUAGE == 'LATVIAN':
        examples += f'<piemērs>\nTeksts: "{section_text}"\n{PROMPT_TYPE.capitalize()}: "{gtc}"\n</piemērs>\n'
    else:
        examples += f'<example>\nText: "{section_text}"\n{PROMPT_TYPE.capitalize()}: "{gtc}"\n</example>\n'

    return section_numbers, examples
# }}}

def load_and_predict_IIR():
# {{{
    print('--- Getting predictions ---')

    # section_number, section_name, section_text, source_url
    with open(DS_JSON_PATH, 'r') as f:
        iir_sections_obj = json.load(f)

    predictions = []
    iir_gt_paths = get_IIR_ground_truth_paths()

    for section, gt_path in zip(iir_sections_obj, iir_gt_paths):
    # {{{
        section_number = section['section_number']
        
        if section_number in SECTIONS or SECTIONS == [0]:
            section_name = section['section_name']
            section_text = section['section_text']
            
            if LANGUAGE == 'LATVIAN':
                template = generate_template_lv(section_name, section_text) 
                print(template)
            else:
                template = generate_template_en(section_name, section_text) 

            # print('SECTION_NUMBERS = ', FS_SECTION_NUMBERS)
            # if section_number not in FS_SECTION_NUMBERS:
            print(section_number, section_name)
            prediction = call_openrouter_api(section_name, template)
            prediction = sorted([c.strip() for c in prediction.split(',')])
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

def load_and_predict_DS():
# {{{
    print('--- Getting predictions ---')

    # section_number, section_name, section_text, source_url
    with open(DS_JSON_PATH, 'r') as f:
        sections = json.load(f)

    predictions = []

    for section in sections:
    # {{{
        section_number = section['section_number']
        
        if section_number in SECTIONS or SECTIONS == [0]:
            section_name = section['section_name']
            section_text = section['section_text']
            
            if LANGUAGE == 'LATVIAN':
                template = generate_template_lv(section_name, section_text) 
                # print(template)
            else:
                template = generate_template_en(section_name, section_text) 

            print(section_number, section_name)
            prediction = call_openrouter_api(section_name, template)
            collator = icu.Collator.createInstance(icu.Locale('lv'))
            prediction = sorted([c.strip() for c in prediction.split(',')], key = collator.getSortKey)
            ground_truth = sorted(list(set(section['gtc'])), key = collator.getSortKey)
            # print('prediction = ', prediction)
            # print('ground_truth =', ground_truth)

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

def write_json(predictions, type = '', file_name = ''):
# {{{
    if file_name != '':
        with open(str(OUTPUTS_PATH) + '/' + file_name, 'w') as f:
            json.dump(predictions, f, indent = 2)
    else:
        with open('outputs/' + PATH_PREFIX + type + '.json', 'w') as f:
            json.dump(predictions, f, ensure_ascii = False, indent = 2)
# }}}

def evaluate_DS(path = '', file_name = ''):
# {{{
    print('--- Evaluations ---')
    if path != '':
        predictions_json_path = path
    else:
        predictions_filename = PATH_PREFIX + 'predictions.json'
        predictions_json_path = (OUTPUTS_PATH / predictions_filename).expanduser()
    
    evaluation = []
    count = 0
    section_text_count_total = 0

    gtc_normalized_total = 0
    pc_normalized_total = 0

    prec_total = 0.0
    rec_total = 0.0
    f1_total = 0.0

    sem_prec_total = 0.0
    sem_rec_total = 0.0
    sem_f1_total = 0.0

    with open(predictions_json_path, 'r') as o:
    # {{{
        predictions_obj = json.load(o)

        for prediction in predictions_obj:
            section_number = prediction['section_number']

            if section_number in SECTIONS or SECTIONS == [0]:
                predictions_normalized = set(normalize_words(prediction['predicted_concepts (pc)']))
                ground_truth_normalized = set(normalize_words(prediction['ground_truth_concepts (gtc)']))

                predictions_stemmed = set(stem_words(list(predictions_normalized)))
                ground_truth_stemmed = set(stem_words(list(ground_truth_normalized)))

                pc_normalized_total += len(predictions_normalized)
                gtc_normalized_total += len(ground_truth_normalized)

                prec, rec, f1 = calc_exact_metrics(predictions_stemmed, ground_truth_stemmed)
                prec_sem, rec_sem, f1_sem = calc_semantical_metrics(list(predictions_normalized), list(ground_truth_normalized))

                prec_total += prec
                rec_total += rec
                f1_total += f1

                sem_prec_total += prec_sem
                sem_rec_total += rec_sem
                sem_f1_total += f1_sem

                count += 1
                section_text_count_total += prediction['section_text_count']

                evaluation.append({
                        'section_number': prediction['section_number'],

                        'precision': prec,
                        'recall': rec,
                        'F1': f1,

                        'sem_precision': prec_sem,
                        'sem_recall': rec_sem,
                        'sem_F1': f1_sem,

                        'pc_count': prediction['pc_count'],
                        'pc_count_to_text_len_ratio (%)': prediction['pc_count_to_text_len_ratio (%)'],
                        'gtc_count': prediction['gtc_count'],
                        'gtc_count_to_text_len_ratio (%)': prediction['gtc_count_to_text_len_ratio (%)'],

                        'section_text_count': prediction['section_text_count']
                    })
        # }}}

    pc_total, pc_unique_total = count_concepts(str(predictions_json_path))
    gtc_total, gtc_unique_total = count_concepts(str(predictions_json_path), 'ground_truth')

    evaluation.append({
            'section_number': 'average',

            'precision_M': round(prec_total / count, 3),
            'recall_M': round(rec_total / count, 3),
            'F1_M': round(f1_total / count, 3),

            'sem_precision_M': round(sem_prec_total / count, 3),
            'sem_recall_M': round(sem_rec_total / count, 3),
            'sem_F1_M': round(sem_f1_total / count, 3),

            'pc_total': pc_total,
            'pc_unique_total': pc_unique_total,
            'pc_normalized_total': pc_normalized_total,

            'gtc_total': gtc_total,
            'gtc_unique_total': gtc_unique_total,
            'gtc_normalized_total': gtc_normalized_total,

            'gtc_pc_normalized_total_diff': gtc_normalized_total - pc_normalized_total,

            'section_text_count_total': section_text_count_total
    })

    print(f'Average: p={(prec_total / count):.3f} r={(rec_total / count):.3f} F1:{(f1_total / count):.3f}')
    print(f'Average (sem): p_sem={(sem_prec_total / count):.3f} r_sem={(sem_rec_total / count):.3f} F1_sem:{(sem_f1_total / count):.3f}')

    write_json(evaluation, 'evaluation', file_name)
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
        result.append(cleaned)
        # stemmed = ' '.join(stemmer.stem(w) for w in cleaned.split())
        # result.append(stemmed)

    return result
# }}}

def stem_words(words: list[str]) -> list[str]:
# {{{
    result: list[str] = []

    for word in words:
        if LANGUAGE == 'LATVIAN':
            stemmed = ' '.join(LatvianStemmer.stem(w) for w in word.split())
        else:
            stemmed = ' '.join(STEMMER.stem(w) for w in word.split())
        result.append(stemmed)

    return result
# }}}

def calc_exact_metrics(predicted: set[str], ground_truth: set[str]) -> tuple[float, float, float]:
# {{{
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) else 0.0
    
    return round(precision, 3), round(recall, 3), round(f1, 3)
# }}}

def calc_semantical_metrics(predicted: list[str], ground_truth: list[str]):
# {{{
    if len(ground_truth) > 0:
        # sem_emb_predictions = SIM_MODEL.encode(predicted)
        # sem_emb_gt = SIM_MODEL.encode(ground_truth)
        sem_emb_predictions = SIM_MODEL.encode(predicted, prompt_name = "sts_query")
        sem_emb_gt = SIM_MODEL.encode(ground_truth)

        # predictions (dim = 0) X gt (dim = 1)
        sem = SIM_MODEL.similarity(sem_emb_predictions, sem_emb_gt).clamp(min = 0)

        sem_p = sem.amax(dim = 1).mean().item()
        sem_r = sem.amax(dim = 0).mean().item()
        sem_f1 = (2 * sem_p * sem_r) / (sem_p + sem_r)

        return round(sem_p, 3), round(sem_r, 3), round(sem_f1, 3)
    else:
        return 0.0, 0.0, 0.0
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
# }}}

def count_concepts(path: str, type = 'predicted') -> tuple[int, int]:
# {{{
    all_concepts = 0
    concepts = []
    unique_concepts = set()

    with open(path, 'r') as o:
        predictions_obj = json.load(o)

        for prediction in predictions_obj:
            if type == 'predicted':
                concepts = prediction['predicted_concepts (pc)']
            elif type == 'ground_truth':
                concepts = prediction['ground_truth_concepts (gtc)']
            all_concepts += len(concepts)
            unique_concepts.update(concepts)

    return all_concepts, len(unique_concepts)
# }}}

def print_parameters():
# {{{
    print()
    print('MODEL = ' + MODEL)
    print('MAX_TOKENS = ' + MAX_TOKENS)
    print('PROMPT_METHOD = ' + PROMPT_METHOD)
    print('PROMPT_TYPE = ' + PROMPT_TYPE)
    print('SHOT_AMOUNT = ' + SHOT_AMOUNT)
    print('SHOT_TYPE = ' + SHOT_TYPE)
    print('TIMESTAMP = ' + TIMESTAMP)
    print('DOMAIN_CONTEXT = ' + DOMAIN_CONTEXT)
    print('CONCEPT_DEFINITION = ' + CONCEPT_DEFINITION)
    print('SYSTEM PROMPT = ' + SYSTEM_PROMPT)
    print('NORMALIZATION = ' + NORMALIZATION)
    print('LANGUAGE = ' + LANGUAGE)
    print('CONSENSUS = ' + CONSENSUS)
    print()
# }}}

def fix_rounding_errors(folder_path: Path):
# {{{
    pattern = r'.*predictions.json'

    for file in folder_path.iterdir():
        reg = re.match(pattern, file.name)

        if reg:
            file_path = str(folder_path) + '/' + file.name
            # print(file.name)
            # print(file_path)
            # print(str(OUTPUTS_PATH) + '/' + file.name)
            evaluate_DS(file_path, file.name)
# }}}

def make_iir_sections_full_file():
# {{{
    # section_number, section_name, section_text, source_url
    with open(DS_JSON_PATH, 'r') as f:
        iir_sections_obj = json.load(f)

    iir_gt_paths = get_IIR_ground_truth_paths()
    obj = []

    for section in iir_sections_obj:
       for path in iir_gt_paths:
        if 'iir-' + section['section_number'] + '.csv' in path:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                threshold = {'ONE': 1, 'TWO': 2, 'THREE': 3}[CONSENSUS]
                ground_truth_concepts = []

                for row in reader:
                    if row[1] and row[2] and row[3]:
                        votes = [row[1][0], row[2][0], row[3][0]].count('1')
                        # Annotator agreement (1)   
                        if votes >= threshold:
                            concept = ast.literal_eval(row[0])[0]
                            ground_truth_concepts.append(concept)

            obj.append({
                'section_number': section['section_number'],
                'section_name': section['section_name'],
                'section_text': section['section_text'],
                'gtc': ground_truth_concepts,
                'source_url': section['source_url'],
            })

    write_json(obj, '', 'iir_sections_full.json')
# }}}

# --- Paths ---
# DS_JSON_PATH = Path('~/Downloads/prereq/scripts/IIR/iir_sections_full.json').expanduser()
DS_JSON_PATH = Path('~/Downloads/prereq/scripts/CE-Books-LV/outputs/bio_vsk_mg.json').expanduser()
IIR_FOLDER_PATH = Path('~/Downloads/prereq/datasets/IIR-dataset/annotation').expanduser()
OUTPUTS_PATH = Path('~/Downloads/prereq/scripts/IIR/outputs/').expanduser()

# --- Models ---
# MODEL = 'claude-haiku-4-5'

# MODEL = ''
# MODELS = ['google/gemma-3-27b-it', 'xiaomi/mimo-v2-flash', 'deepseek/deepseek-v3.2', 'x-ai/grok-4.1-fast', 'moonshotai/kimi-k2.5', 'z-ai/glm-4.7', 'google/gemini-3-flash-preview']
# MODELS = ['xiaomi/mimo-v2-flash', 'deepseek/deepseek-v3.2', 'x-ai/grok-4.1-fast', 'moonshotai/kimi-k2.5', 'z-ai/glm-4.7', 'google/gemini-3-flash-preview']
MODELS = ['moonshotai/kimi-k2.5', 'z-ai/glm-4.7', 'google/gemini-3-flash-preview']
# MODELS = ['x-ai/grok-4.1-fast', 'z-ai/glm-4.7', 'google/gemini-3-flash-preview']
# MODELS = ['google/gemma-3-27b-it']

# MODEL = 'openai/gpt-oss-120b'
# MODEL = 'xiaomi/mimo-v2-flash'
# MODEL = 'deepseek/deepseek-v3.2'
# MODEL = 'x-ai/grok-4.1-fast'
# MODEL = 'moonshotai/kimi-k2.5'
# MODEL = 'z-ai/glm-4.7'
# MODEL = 'google/gemma-3-27b-it'
# MODEL = 'google/gemini-3-flash-preview'

# MODEL = 'openai/gpt-5-mini' # Mandatory reasoning 
# MODEL = 'minimax/minimax-m2.5' # Mandatory reasoning

# Static
MAX_TOKENS = '1024'
SYSTEM_PROMPT = 'SYSTEM-PROMPT-YES'
CONSENSUS = 'THREE'
NORMALIZATION = 'STEMMED'
LANGUAGE = 'LATVIAN'

# --- Methods ---
PROMPT_METHOD = 'ZERO-SHOT'
# PROMPT_METHOD = 'FEW-SHOT'

# SHOT_AMOUNT = ''
SHOT_AMOUNT = 'ZERO'
# SHOT_AMOUNT = 'FIVE-SHOT'
# SHOT_AMOUNTS = ['ONE-SHOT', 'THREE-SHOT', 'FIVE-SHOT']

# SHOT_TYPE = ''
SHOT_TYPE = 'ZERO'
# SHOT_TYPE = 'RANDOM'
# SHOT_TYPES = ['FIRST', 'RANDOM']

# PROMPT_TYPE = 'CONCEPTS'
# PROMPT_TYPE = 'KEYWORDS'
# PROMPT_TYPE = 'KEYPHRASES'
# PROMPT_TYPE = 'TERMS'
# PROMPT_TYPE = 'TERMINI'
PROMPT_TYPE = ''
PROMPT_TYPES = ['KONCEPTI', 'ATSLĒGVĀRDI', 'GALVENĀS FRĀZES', 'TERMINI']
# PROMPT_TYPES = ['CONCEPTS', 'KEYWORDS', 'KEYPHRASES', 'TERMS']
# PROMPT_TYPES = ['CONCEPTS', 'TERMS']

DOMAIN_CONTEXT = 'DOMAIN-CONTEXT-YES'
# DOMAIN_CONTEXT = 'DOMAIN-SUBCONTEXT-YES'
# DOMAIN_CONTEXT = 'DOMAIN-CONTEXT-NO'
# DOMAIN_CONTEXT = ''
# DOMAIN_CONTEXTS = ['DOMAIN-CONTEXT-YES', 'DOMAIN-SUBCONTEXT-YES']

CONCEPT_DEFINITION = 'CONCEPT-DEFINITION-NO'
# CONCEPT_DEFINITION = 'CONCEPT-DEFINITION-KEY'
# CONCEPT_DEFINITION = 'CONCEPT-DEFINITION-YES'
# CONCEPT_DEFINITION = 'CONCEPT-DEFINITION-WIKIPEDIA'
# CONCEPT_DEFINITION = ''
# CONCEPT_DEFINITIONS = ['CONCEPT-DEFINITION-YES', 'CONCEPT-DEFINITION-WIKIPEDIA']

# --- Timestamps ---
# TIMESTAMP = '2026-04-20_04-49-06' # For testing

# --- Sections ---
# SECTIONS = ['4.4']
# SECTIONS = ['1']
# SECTIONS = ['1', '1.1', '1.2', '1.3', '1.4']
SECTIONS = [0] # All sections

# --- Other ---
start = time.time()

if LANGUAGE == 'LATVIAN':
    # SIM_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    SIM_MODEL = SentenceTransformer('microsoft/harrier-oss-v1-0.6b', model_kwargs={'dtype': 'auto'})
else:
    STEMMER = SnowballStemmer('english')
    SIM_MODEL = SentenceTransformer('uclanlp/keyphrase-mpnet-v1')

TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
load_dotenv()
global CLIENT
CLIENT = make_openai_client()

# ONE-SHOT RUN
# {{{
for i in range (len(MODELS)):
    MODEL = MODELS[i]

    for j in range (len(PROMPT_TYPES)):
        PROMPT_TYPE = PROMPT_TYPES[j]
        TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        PATH_PREFIX = MODEL.replace('/', '-') + '_' + MAX_TOKENS + '_' + PROMPT_METHOD + '_' + SHOT_AMOUNT + '_' + SHOT_TYPE + '_' + PROMPT_TYPE + '_' + LANGUAGE + '_' + DOMAIN_CONTEXT + '_' + CONCEPT_DEFINITION + '_' + SYSTEM_PROMPT + '_' + CONSENSUS + '-CONSENSUS_' + NORMALIZATION + '_' + TIMESTAMP + '_'

        print_parameters()
        load_and_predict_DS()
        evaluate_DS()
# }}}

# FEW-SHOT RUN
# {{{
# for i in range(3):
# TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# FS_SECTION_NUMBERS, EXAMPLES = get_few_shot_examples()
# PATH_PREFIX = MODEL.replace('/', '-') + '_' + MAX_TOKENS + '_' + PROMPT_METHOD + '_' + SHOT_AMOUNT + '_' + SHOT_TYPE + '_' + PROMPT_TYPE + '_' + LANGUAGE + '_' + DOMAIN_CONTEXT + '_' + CONCEPT_DEFINITION + '_' + SYSTEM_PROMPT + '_' + CONSENSUS + '-CONSENSUS_' + NORMALIZATION + '_' + TIMESTAMP + '_'

# print_parameters()
# load_and_predict_IIR() # Makes API calls
# load_and_predict_DS()
# evaluate_DS('/home/dust/Downloads/prereq/scripts/IIR/outputs/google-gemma-3-27b-it_1024_FEW-SHOT_FIVE-SHOT_RANDOM_TERMINI_LATVIAN_DOMAIN-CONTEXT-YES_CONCEPT-DEFINITION-NO_SYSTEM-PROMPT-YES_THREE-CONSENSUS_STEMMED_2026-04-27_18-56-56_predictions.json')
# print(generate_template_lv('HELLO', 'WORLD'))
# }}}


elapsed = time.time() - start
print(f'\nTotal elapsed time: {elapsed:.2f}s')

