import ast
import traceback
import logging
import re
import math
import inflect
from roman import fromRoman  # Convert Roman numerals back to integers
from nltk.translate.bleu_score import sentence_bleu
from word2number import w2n
from collections import defaultdict
import datetime
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def normalize_tabfact_prediction(pred):
    """
    Normalize model output to:
    True / False / None
    """
    pred = extract_text_after_tag(pred)
    pred = pred.strip().lower()

    # Strip punctuation
    pred_clean = re.sub(r"[^\w\s]", " ", pred)
    pred_clean = re.sub(r"\s+", " ", pred_clean).strip()

    true_keywords = [
        "true",
        "support",
        "supported",
        "supports",
        "entailed",
        "entailment",
        "yes",
        "correct",
    ]

    false_keywords = [
        "false",
        "refute",
        "refuted",
        "refutes",
        "not supported",
        "unsupported",
        "contradicted",
        "contradiction",
        "no",
        "incorrect",
    ]

    # Prefer matching whole words
    for kw in false_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", pred_clean):
            return "False"

    for kw in true_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", pred_clean):
            return "True"

    # Fallback: substring match for true/false
    if "false" in pred:
        return "False"
    if "true" in pred:
        return "True"

    return None


def normalize_tabfact_gold(gold):
    if isinstance(gold, list):
        gold = gold[0] if gold else ''
    if isinstance(gold, str):
        gold = gold.strip()
        if gold.lower() == 'true':
            return 'True'
        if gold.lower() == 'false':
            return 'False'
    return gold


def cal_tabfact_score(pred, gold):
    """
    pred: raw LLM output
    gold: 'True' or 'False'

    return:
        1 -> correct
        0 -> incorrect
    """
    gold = normalize_tabfact_gold(gold)
    pred_label = normalize_tabfact_prediction(pred)

    if pred_label is None:
        return 0, 0, 0

    correct = int(pred_label == gold)
    return correct, correct, correct


def cal_tabfact_aggregate_metrics(items):
    """
    Aggregate TabFact metrics from a list of samples.

    return:
        overall_acc: overall accuracy
        true_acc: #[pred True, gold True] / #[gold True]
        false_acc: #[pred False, gold False] / #[gold False]
    """
    if not items:
        return 0.0, 0.0, 0.0

    correct = 0
    gold_true = 0
    pred_true_gold_true = 0
    gold_false = 0
    pred_false_gold_false = 0

    for item in items:
        raw_output = item.get('output_text', '')
        gold = item.get('gold_answer', item.get('answer'))
        gold = normalize_tabfact_gold(gold)
        pred_label = normalize_tabfact_prediction(raw_output)

        if pred_label is not None and pred_label == gold:
            correct += 1

        if gold == 'True':
            gold_true += 1
            if pred_label == 'True':
                pred_true_gold_true += 1
        elif gold == 'False':
            gold_false += 1
            if pred_label == 'False':
                pred_false_gold_false += 1

    n = len(items)
    overall_acc = correct / n
    true_acc = pred_true_gold_true / gold_true if gold_true > 0 else 0.0
    false_acc = pred_false_gold_false / gold_false if gold_false > 0 else 0.0
    return overall_acc, true_acc, false_acc

def normalize_text(text: str) -> str:
    """
    Basic normalization to reduce BLEU noise.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)          # Collapse whitespace
    text = re.sub(r"[^\w\s]", "", text)      # Strip punctuation (optional)
    return text.strip()

def token_f1(pred_tokens, gold_tokens):
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)

def cal_bleu(pred: str, gold: str):

    # 1. Extract the final answer
    pred = extract_text_after_tag(pred, tag="</think>")
    pred = pred.strip()

    # 2. Normalize
    pred = normalize_text(pred)
    gold = normalize_text(gold)

    # 3. Tokenization
    pred_tokens = pred.split()
    gold_tokens = gold.split()

    # 4. Guard against empty outputs
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0, 0.0, 0.0

    # ===== BLEU =====
    bleu = sentence_bleu(
        [gold_tokens],
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1
    )

    # ===== ROUGE-L =====
    rouge_l = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    ).score(gold, pred)["rougeL"].fmeasure

    # ===== Token F1 =====
    f1 = token_f1(pred_tokens, gold_tokens)

    return float(bleu), float(rouge_l), float(f1)

def cal_bleu_old(pred: str, gold: str):
    """
    Input:
        pred: raw LLM output (may contain <think>)
        gold: reference answer (str)
    Output:
        BLEU score (float)
    """

    # 1. Extract the final answer
    pred = extract_text_after_tag(pred, tag="</think>")

    # 2. Normalize
    pred = normalize_text(pred)
    gold = normalize_text(gold)

    # 3. Tokenization
    pred_tokens = pred.split()
    gold_tokens = [gold.split()]  # reference must be a list of references

    # 4. Guard against empty outputs
    if len(pred_tokens) == 0 or len(gold_tokens[0]) == 0:
        return 0.0

    # 5. Smoothing (important for short texts)
    smoothie = SmoothingFunction().method1

    bleu = sentence_bleu(
        gold_tokens,
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )

    return float(bleu),float(bleu),float(bleu)

def english_to_number_old(word):
    p = inflect.engine()
    try:
        # Try converting an English word back to a number
        result=p.parse(word)[0][1]
        return result  # Return the numeric part
    except:
        return word  # If conversion fails, return the original

def scientific_to_number(sci_str):
    # Convert scientific notation (e.g. 1.23×10^5) back to a normal number
    match = re.match(r"([+-]?\d*\.\d+)×10\^([+-]?\d+)", sci_str)
    if match:
        base = float(match.group(1))
        exponent = int(match.group(2))
        return base * (10 ** exponent)
    return sci_str  # If no match, return the original string

def english_to_number(word):
    # Do not convert date-like strings
    if '.' in word:
        return word
    try:
        return w2n.word_to_num(word)
    except:
        return word

def roman_to_number(roman_str):
    try:
        # Convert lowercase Roman numerals
        return fromRoman(roman_str.upper())  # Normalize to uppercase before conversion
    except:
        return roman_str  # If conversion fails, return the original

# Slightly different return value from roman_to_number
def convert_from_roman(roman):
    try:
        return fromRoman(roman)
    except:
        return None

def reverse_process_string(value):
    # Detect and reverse-transform
    if isinstance(value, str):
        if re.match(r'^[ivxlcxdm]+$', value):  # Roman numeral
            return roman_to_number(value)
        elif '×10^' in value:  # Scientific notation
            return scientific_to_number(value)
        else:  # Otherwise treat as an English word
            return english_to_number(value)
    return value  # If not a string, return as-is

# Convert a modified date string back to a datetime object
def parse_modified_date(date_str):
    try:
        # Try parsing Roman-numeral date format
        roman_pattern = r"^(.*)-(.*)-(.*)$"
        roman_match = re.match(roman_pattern, date_str)
        if roman_match:
            year = convert_from_roman(roman_match.group(1))
            month = convert_from_roman(roman_match.group(2))
            day = convert_from_roman(roman_match.group(3))
            if year is not None and month is not None and day is not None:
                return datetime.datetime(year, month, day)

        # Try parsing English-word date format
        words_pattern = r"^(.*)\.(.*)\.(.*)$"
        words_match = re.match(words_pattern, date_str)
        if words_match:
            year = english_to_number(words_match.group(1))
            month = english_to_number(words_match.group(2))
            day = english_to_number(words_match.group(3))
            if year is not None and month is not None and day is not None:
                return datetime.datetime(year, month, day)

        # Try parsing standard date formats
        date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m.%d.%Y"]
        for fmt in date_formats:
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except:
                continue

        return date_str
    except:
        return data_str

# Check whether two strings are equal (including date variants)
def is_equal_datetime(str1, str2):
    # Condition 2.1: equal after lowercasing
    if str1.lower() == str2.lower():
        return True

    # Condition 2.2: both parse as datetime and are equal
    date1 = parse_modified_date(str1)
    date2 = parse_modified_date(str2)
    #print(date1,date2)
    if date1 is not None and date2 is not None and date1 == date2:
        return True

    # Condition 2.3: otherwise not equal
    return False


def calculate_bleu(reference: str, candidate: str) -> float:
    reference_split = reference.split()  # Tokenize the reference
    candidate_split = candidate.split()  # Tokenize the candidate
    bleu_score = sentence_bleu([reference_split], candidate_split)
    return bleu_score

def cal_score(reference, candidate):
    # print(candidate.lower())
    # print(reference.lower())
    reference = str(reference)
    if candidate.lower() == reference.lower():
        # print("1")
        return 1
    # if candidate.lower() in reference.lower():
    #    print("1")
    #    return 1
    return calculate_bleu(reference, candidate)

def is_list_of_strings(variable):
    if isinstance(variable, list) and all(isinstance(item, str) for item in variable):
        return True
    return False


def convert_string(A: str):
    try:
        result = ast.literal_eval(A)
        if isinstance(result, str):  # Extra handling
            return [result]
        if isinstance(result, list):
            # If it is a list, ensure each element is a string
            return [str(i) if not isinstance(i, str) else i for i in result]
    except Exception as e:
        error_message = traceback.format_exc()
        #logging.info(f"An error occurred: {error_message}")
        pass
    return [A]


def trans_unsupport(input_str):
    # Patterns to match
    pattern1 = r"\{'Unsupported'\}"  # Match {'Unsupported'}
    pattern2 = r'\{"Unsupported"\}'  # Match {"Unsupported"}
    pattern3 = r"\{'Unsupported': 'Unsupported'\}"  # Match {'Unsupported': 'Unsupported'}
    pattern4 = r'\{"Unsupported": "Unsupported"\}'  # Match {"Unsupported": "Unsupported"}
    pattern5 = r"\{'result': 'Unsupported'\}"  # Match {'result': 'Unsupported'}
    pattern6 = r'\{"result": "Unsupported"\}"'  # Match {'result': 'Unsupported'}

    # Case-insensitive match for any of the patterns
    if (re.search(pattern1, input_str, re.IGNORECASE) or
            re.search(pattern2, input_str, re.IGNORECASE) or
            re.search(pattern3, input_str, re.IGNORECASE) or
            re.search(pattern4, input_str, re.IGNORECASE) or
            re.search(pattern5, input_str, re.IGNORECASE) or
            re.search(pattern6, input_str, re.IGNORECASE)):

        # Replace all matches with ['Unsupported']
        input_str = re.sub(pattern1, '["Unsupported"]', input_str, flags=re.IGNORECASE)
        input_str = re.sub(pattern2, '["Unsupported"]', input_str, flags=re.IGNORECASE)
        input_str = re.sub(pattern3, '["Unsupported"]', input_str, flags=re.IGNORECASE)
        input_str = re.sub(pattern4, '["Unsupported"]', input_str, flags=re.IGNORECASE)
        input_str = re.sub(pattern5, '["Unsupported"]', input_str, flags=re.IGNORECASE)
        input_str = re.sub(pattern6, '["Unsupported"]', input_str, flags=re.IGNORECASE)

        return input_str
    else:
        # No match: return the original string
        return input_str

def extract_json_content2(s):
    match = re.search(r'```json(.*?)```', s, re.DOTALL)
    if match:
        return match.group(1).strip()  # Extract and strip surrounding whitespace
    return s  # If no match, return the original string

def extract_json_content(s):
    matches = re.findall(r'```json(.*?)```', s, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last match, stripped
    return s

def extract_python_content(s):
    matches = re.findall(r'```python(.*?)```', s, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last match, stripped
    return s

def extract_list_content(s):
    matches = re.findall(r'```list(.*?)```', s, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last match, stripped
    return s

def extract_python_dict_content(s):
    matches = re.findall(r'```python\nlist[dict]:(.*?)```', s, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last match, stripped
    return s

def extract_granite(text,tag='list[str]:'):
    return text.split(tag)[-1].strip()

def extrcat_tulu(s):
    # Find the last occurrence of "list:"
    last_index = s.rfind("list:")

    # If "list:" was found
    if last_index != -1:
        # Return content after the last "list:", stripped
        return s[last_index + len("list:"):].strip()

    # If "list:" was not found, return the whole string stripped
    return s.strip()

def extract_tablellm(input_str):
    # Case-insensitive match for "answer_item:", "answer:", or "list[dict]:"
    pattern = re.compile(r'(answer_item:|answer:|list\[dict\]:)', re.IGNORECASE)
    matches = pattern.finditer(input_str)

    # Find the end position of the last match
    last_match_end = None
    for match in matches:
        last_match_end = match.end()

    # If a match was found
    if last_match_end is not None:
        # Return the suffix after the last match, stripped
        return input_str[last_match_end:].strip()

    # If no pattern matched, but the string starts with "list["
    if input_str.lower().startswith("list["):
        # Return the part after "list" (including "[")
        return input_str[4:].strip()

    # Otherwise return the original string
    return input_str

def extract_tablellama(s):
    pattern = r"^<(.*?)>(?:, <(.*?)>)*$"
    matches = re.findall(r"<([^<>]+)>", s)

    if matches:
        return "['" + "', '".join(matches) + "']"
    return s

def extract_three(input_str):
    # Count occurrences of "```"
    count = input_str.count('```')

    # If "```" appears exactly once
    if count == 1:
        index = input_str.find('```')

        # If "```" is at the end of the string
        if index + 3 == len(input_str):
            return input_str[:index]
        else:
            return input_str[index + 3:]

    # Otherwise return the original string
    return input_str

def extract_sql_content(s):
    matches = re.findall(r'```sql(.*?)```', s, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last match, stripped
    return s

def extract_ori_content(s):
    matches = re.findall(r'```(.*?)```', s, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last match, stripped
    return s

def extract_text_after_tag(text, tag="</think>"):
    return text.split(tag)[-1].strip()

def cal_score_list_old(gold, pred):
    #if is_list_of_strings(gold) == False:
    #    gold = [str(gold)]
    pred = convert_string(pred)
    #if is_list_of_strings(pred) == False:
    #    pred = [str(pred)]
    # Step 1: Strip whitespace/newlines and lowercase
    gold = [str(item) for item in gold]
    gold = [item.strip().lower() for item in gold]
    pred = [item.strip().lower() for item in pred]
    print(pred)
    #print(gold)
    #print("pred",pred)

    # Step 2: Compute precision, recall, and F1
    # Convert list1/list2 to sets: intersection = TP,
    # in list2 but not list1 = FP, in list1 but not list2 = FN
    set1, set2 = set(gold), set(pred)
    tp = len(set1 & set2)  # True Positive
    fp = len(set2 - set1)  # False Positive
    fn = len(set1 - set2)  # False Negative

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def normalize_item(item):
    """Normalize invalid items (None, NaN, or string forms of Nan/None) to None."""
    if item is None:
        return None
    if isinstance(item, float) and math.isnan(item):
        return None
    if isinstance(item, str) and item.lower() in ['nan', 'none']:
        return None
    return item

def cal_score_list(gold, pred):
    # Step 1: Infer gold type and convert pred
    # Convert gold and pred to string lists (further normalized below)
    gold = [str(item) for item in gold]
    pred = convert_string(pred)

    # Convert values so subsequent comparisons work correctly
    def safe_convert(value):
        if isinstance(value, str):
            # Check for "null", "nan", etc.
            if value.lower() in ['null', 'nan', 'none']:
                return None
            # Try converting to a number
            try:
                if '.' in value:  # Likely a float
                    return float(value.strip())
                else:  # Likely an integer
                    return int(value.strip())
            except ValueError:
                # On failure, keep as a lowercased string
                return value.strip().lower()
        if isinstance(value, (int, float)):
            return value
        return str(value).strip().lower()

    # Safely convert each element in pred/gold
    pred = [safe_convert(item) for item in pred]
    gold = [safe_convert(item) for item in gold]
    pred = [reverse_process_string(item) for item in pred]
    gold = [reverse_process_string(item) for item in gold]

    # Normalize invalid items to None
    list1_normalized = [normalize_item(item) for item in gold]
    list2_normalized = [normalize_item(item) for item in pred]

    # Assume list1_normalized and list2_normalized are already processed
    list1_counts = defaultdict(int)
    for item in list1_normalized:
        list1_counts[item] += 1

    # Traverse list2 to compute TP and FP
    tp = 0
    fp = 0

    for item in list2_normalized:
        matched = False
        for key in list(list1_counts.keys()):
            # If both are strings, compare with is_equal_datetime
            if isinstance(item, str) and isinstance(key, str):
                if is_equal_datetime(item, key) and list1_counts[key] > 0:
                    tp += 1
                    list1_counts[key] -= 1
                    matched = True
                    break
            # Otherwise compare with ==
            elif item == key and list1_counts[key] > 0:
                tp += 1
                list1_counts[key] -= 1
                matched = True
                break

        if not matched:
            fp += 1

    # Remaining counts in list1_counts are FN
    fn = sum(list1_counts.values())

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def cal_score_list2(gold, pred):
    # Step 1: Infer gold type and convert pred
    # Convert gold and pred to string lists (further normalized below)
    gold = [str(item) for item in gold]
    pred = convert_string(pred)

    # Convert values so subsequent comparisons work correctly
    def safe_convert(value):
        if isinstance(value, str):
            # Check for "null", "nan", etc.
            if value.lower() in ['null', 'nan', 'none']:
                return None
            # Try converting to a number
            try:
                if '.' in value:  # Likely a float
                    return float(value.strip())
                else:  # Likely an integer
                    return int(value.strip())
            except ValueError:
                # On failure, keep as a lowercased string
                return value.strip().lower()
        if isinstance(value, (int, float)):
            return value
        return str(value).strip().lower()

    # Safely convert each element in pred/gold
    pred = [safe_convert(item) for item in pred]
    gold = [safe_convert(item) for item in gold]
    pred = [reverse_process_string(item) for item in pred]
    gold = [reverse_process_string(item) for item in gold]
    #print(pred)
    #print(gold)

    # Step 2: Compute precision, recall, and F1
    # Convert list1/list2 to sets: intersection = TP,
    # in list2 but not list1 = FP, in list1 but not list2 = FN
    set1, set2 = set(gold), set(pred)

    # Exclude NaN/None so invalid items are ignored
    set1 = {item for item in set1 if item not in [None, float('nan')]}
    set2 = {item for item in set2 if item not in [None, float('nan')]}

    tp = len(set1 & set2)  # True Positive
    fp = len(set2 - set1)  # False Positive
    fn = len(set1 - set2)  # False Negative

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def cal_score_list3(gold, pred):
    # Step 1: Infer gold type and convert pred
    # Convert gold and pred to string lists (further normalized below)
    gold = [str(item) for item in gold]
    pred = convert_string(pred)

    # Convert values so subsequent comparisons work correctly
    def safe_convert(value):
        if isinstance(value, str):
            # Check for "null", "nan", etc.
            if value.lower() in ['null', 'nan', 'none']:
                return None
            # Try converting to a number
            try:
                if '.' in value:  # Likely a float
                    return float(value.strip())
                else:  # Likely an integer
                    return int(value.strip())
            except ValueError:
                # On failure, keep as a lowercased string
                return value.strip().lower()
        if isinstance(value, (int, float)):
            return value
        return str(value).strip().lower()

    # Safely convert each element in pred/gold
    pred = [safe_convert(item) for item in pred]
    gold = [safe_convert(item) for item in gold]
    pred = [reverse_process_string(item) for item in pred]
    gold = [reverse_process_string(item) for item in gold]

    # Normalize invalid items to None
    list1_normalized = [normalize_item(item) for item in gold]
    list2_normalized = [normalize_item(item) for item in pred]

    # Compute TP, FP, FN
    tp = 0
    fp = 0
    fn = 0

    # Count occurrences of each element in list1
    from collections import defaultdict
    list1_counts = defaultdict(int)
    for item in list1_normalized:
        list1_counts[item] += 1

    # Traverse list2 to compute TP and FP
    for item in list2_normalized:
        if item in list1_counts and list1_counts[item] > 0:
            tp += 1
            list1_counts[item] -= 1
        else:
            fp += 1

    # Remaining counts in list1_counts are FN
    fn = sum(list1_counts.values())

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def convert_to_dict(input_str):
    old_string = input_str
    try:
        # Lowercase the string
        input_str = input_str.lower()

        # Replace 'true'/'false' with Python booleans
        input_str = input_str.replace('true', 'True')
        input_str = input_str.replace('false', 'False')

        # Replace 'nan' with placeholder 'NNANN' (uppercase so it will not collide)
        input_str = input_str.replace('"nan"', '\'NNANN\'')
        input_str = input_str.replace('\'nan\'', '\'NNANN\'')
        input_str = re.sub(r'(?<=[ ,:}])nan(?=[ ,:}])', '\'NNANN\'', input_str)
        input_str = input_str.replace('"none"', '\'NNANN\'')
        input_str = input_str.replace('\'none\'', '\'NNANN\'')
        input_str = re.sub(r'(?<=[ ,:}])none(?=[ ,:}])', '\'NNANN\'', input_str)
        #print(input_str)
        # Parse with ast.literal_eval
        result = ast.literal_eval(input_str)

        # If it is a dict, replace 'NNANN' with float('nan')
        if isinstance(result, dict):
            for key, value in result.items():
                if value == "NNANN":
                    result[key] = float('nan')

        # Return the result
        return result if isinstance(result, dict) else None
    except Exception as e:
        error_message = traceback.format_exc()
        #if 'unsupported' not in old_string.lower():
        #    print(old_string)
        #    print(f"An error occurred: {error_message}")
        return None

def compare_dicts_ignore_case(dict1, dict2):
    # If lengths differ, they cannot be equal
    if len(dict1) != len(dict2):
        return False

    # Helper: check whether values are NaN or empty (None)
    def is_equivalent_value(value1, value2):
        # Treat None, NaN, and empty string as equivalent
        def is_equivalent(v):
            return v is None or (isinstance(v, float) and math.isnan(v)) or v == ''

        # Both are equivalent null-like values
        if is_equivalent(value1) and is_equivalent(value2):
            return True

        # Both strings: compare with is_equal_datetime
        if isinstance(value1, str) and isinstance(value2, str):
            return is_equal_datetime(value1, value2)

        # Otherwise compare directly
        return value1 == value2

    # Iterate over each key/value
    for key in dict1:
        # Compare keys in lowercase
        lower_key = key.lower()

        value1 = dict1[key]
        value2 = dict2.get(lower_key, None)
        #print(value1,value2)

        # Lowercase and reverse-transform string values before comparison
        if isinstance(value1, str):
            value1 = value1.lower()
            value1 = reverse_process_string(value1)
        if isinstance(value2, str):
            value2 = value2.lower()
            value2 = reverse_process_string(value2)

        #print(value1, value2)

        # Attempt type coercion when needed
        if isinstance(value1, float):
            try:
                value2 = float(value2) if value2 is not None else value2
            except (ValueError, TypeError):
                pass
        elif isinstance(value1, int):
            try:
                value2 = int(value2) if value2 is not None else value2
            except (ValueError, TypeError):
                pass
        elif isinstance(value1, bool):
            value2 = bool(value2) if value2 is not None else value2

        # Compare with the helper
        if not is_equivalent_value(value1, value2):
            return False

    return True

def compare_dicts_ignore_case2(dict1, dict2):
    # If lengths differ, they cannot be equal
    if len(dict1) != len(dict2):
        return False

    # Helper: check whether values are NaN or empty (None)
    def is_equivalent_value(value1, value2):
        # Treat None, NaN, and empty string as equivalent
        def is_equivalent(v):
            return v is None or (isinstance(v, float) and math.isnan(v)) or v == ''

        # Both are equivalent null-like values
        if is_equivalent(value1) and is_equivalent(value2):
            return True

        # Otherwise compare directly
        return value1 == value2

    # Iterate over each key/value
    for key in dict1:
        # Compare keys in lowercase
        lower_key = key.lower()

        value1 = dict1[key]
        value2 = dict2.get(lower_key, None)

        # Lowercase and reverse-transform string values before comparison
        if isinstance(value1, str):
            value1 = value1.lower()
            value1 = reverse_process_string(value1)
        if isinstance(value2, str):
            value2 = value2.lower()
            value2 = reverse_process_string(value2)

        # Attempt type coercion when needed
        if isinstance(value1, float):
            try:
                value2 = float(value2) if value2 is not None else value2
            except (ValueError, TypeError):
                pass
        elif isinstance(value1, int):
            try:
                value2 = int(value2) if value2 is not None else value2
            except (ValueError, TypeError):
                pass
        elif isinstance(value1, bool):
            value2 = bool(value2) if value2 is not None else value2

        # Compare with the helper
        if not is_equivalent_value(value1, value2):
            return False

    return True

def cal_score_dict(gold, pred):
    ori_pred=pred
    pred=convert_to_dict(pred)
    if pred is None:
        #print("gold",gold)
        #print("pred",ori_pred)
        return 0,0,0
    if compare_dicts_ignore_case(gold, pred)==True:
        return 1,1,1
    return 0,0,0

def get_text_after_double_newline(s):
    # Find the last occurrence of two consecutive newlines
    double_newline_index = s.rfind('\n\n')

    # If found
    if double_newline_index != -1:
        # Take the suffix after the double newline
        result = s[double_newline_index + 2:]
        # Strip surrounding whitespace/newlines
        return result.strip()

    # If not found, return the original string
    return s


def extract_granite_response(text):
    pattern = r'<response>(.*?)</response>'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    return text


def get_content_command(input_string: str) -> str:
    # Find the index of '<|END_RESPONSE|>'
    end_index = input_string.find('<|END_RESPONSE|>')

    # If found, return the prefix before it
    if end_index != -1:
        return input_string[:end_index]
    else:
        # Otherwise return the original string
        return input_string


def extract_answer(text):
    """
    Extract the string between <answer> and </answer>;
    if there is no match, return the original string.

    Args:
        text (str): Input string to process.

    Returns:
        str: Matched content or the original string.
    """
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text

def cal_score3(gold, pred):
    try:
        pred = get_content_command(pred)
        pred = extract_answer(pred)
        pred = extract_text_after_tag(pred)
        pred = extract_text_after_tag(pred,'◁/think▷')
        pred = extract_granite_response(pred)
        pred = get_text_after_double_newline(pred)
        pred = trans_unsupport(pred)
        pred = extract_tablellama(pred)
        pred = extract_tablellm(pred)
        pred = extrcat_tulu(pred)
        pred = extract_python_dict_content(pred)
        pred = extract_granite(pred)
        pred = extract_json_content(pred)
        pred = extract_python_content(pred)
        pred = extract_list_content(pred)
        pred = extract_ori_content(pred)
        pred = extract_three(pred)
        if isinstance(gold, dict):
            return cal_score_dict(gold, pred)
        else:
            return cal_score_list(gold, pred)
    except:
        return 0,0,0

if __name__ == '__main__':
    gold=['Unsupported']
    pred='\"Unsupported\"'
    score = cal_score3(gold, pred)
    print(score)
    gold=[11,22,33,1,2,3,'2023-01-01',123000]
    pred="[11.0,22.0,13.0,12.001,'two','I','two thousand twenty-three.one.one','1.23×10^5']"
    score=cal_score3(gold,pred)
    print(score)
    print(reverse_process_string("one"))  # Expected: 1
    print(reverse_process_string("1.23×10^5"))  # Expected: 123000.0

    dict1 = {"date": "2023-01-01", "value": 100}
    dict2 = {"date": "two thousand twenty-three.one.one", "value": 100}
    print(compare_dicts_ignore_case(dict1,dict2))
