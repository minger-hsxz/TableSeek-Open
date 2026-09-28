"""
Direct TableQA evaluation (gold or retrieved tables).

Given a query JSON and table CSVs, call an LLM to answer short-form TableQA
questions and compute precision / recall / F1.
"""

import ast
import json
import os
import sys
import re
import csv
import logging
import datetime
import warnings
import traceback
import argparse
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
from openai import OpenAI

warnings.filterwarnings("ignore")

_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from tableseek.reformat import df2mk, df2html, df2sql, df2latex, df2xml, df2json, df2csv
from tableseek.metrics import cal_score3
from tableseek.logging_utils import setup_logging


def save_csv(data, path):
    if not data:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def str_to_list(s):
    return ast.literal_eval(s)


def parse_table_key(key):
    db_id, table_path = key.split("::", 1)
    return db_id, table_path


def get_table_keys(example, table_source, top_k):
    if table_source == "gold":
        return [(example["db_id"], example["highlighted_table"][0])]
    if table_source == "dense":
        keys = str_to_list(example["output_text"])[:top_k]
    elif table_source == "rerank":
        keys = str_to_list(example["output_text_rerank"])[:top_k]
    else:
        raise ValueError("Unknown table_source: {}".format(table_source))
    return [parse_table_key(k) for k in keys]


def get_real_table_name(s):
    m = re.match(r"^(.*)_T\d+$", s)
    return m.group(1) if m else s


def trans_tables(table_name, df, table_format):
    table_name = get_real_table_name(table_name)
    if table_format == "markdown":
        table_infos = df2mk(df)
    elif table_format == "html":
        table_infos = df2html(df)
    elif table_format == "sql":
        table_infos = df2sql(df, table_name)
    elif table_format == "latex":
        table_infos = df2latex(df)
    elif table_format == "xml":
        table_infos = df2xml(df)
    elif table_format == "json":
        table_infos = df2json(df)
    elif table_format == "df":
        table_infos = str(df)
    elif table_format == "csv":
        table_infos = df2csv(df)
    else:
        table_infos = df2mk(df)
    table_infos = re.sub(r" +", " ", table_infos)
    return "{}:\n{}\n\n".format(table_name, table_infos)


def find_table_format(table_list, db_id, table_path):
    for item in table_list:
        if item.get("db_id") == db_id and item.get("table_path") == table_path:
            return item.get("table_format", "markdown")
    return "markdown"


def load_single_table(db_id, table_path, tables_path, table_format, table_list):
    db_path = os.path.join(tables_path, db_id)
    table_name = table_path
    files_in_directory = os.listdir(db_path)
    matching_files = [
        f for f in files_in_directory if f.lower() == "{}.csv".format(table_name).lower()
    ]
    table_file_path = os.path.join(db_path, matching_files[0])
    df = pd.read_csv(table_file_path, low_memory=False)
    real_format = (
        find_table_format(table_list, db_id, table_path)
        if table_format == "mixed"
        else table_format
    )
    return trans_tables(table_name, df, real_format)


def load_tables(example, tables_path, table_format, table_list, table_source="gold", top_k=1):
    table_keys = get_table_keys(example, table_source, top_k)
    return "".join(
        load_single_table(db_id, table_path, tables_path, table_format, table_list)
        for db_id, table_path in table_keys
    )


def build_prompt(example, tables_path, table_format, table_list, table_source="gold", top_k=1):
    table_infos = load_tables(
        example, tables_path, table_format, table_list, table_source, top_k
    )
    query = example["question"]
    sys_prompt = (
        "### Requirements:\n"
        "Please read the following table and then answer the questions based on the table.\n"
        "Organize your answers into a list of strings, with each element being an answer item "
        "(The number of answer items is less than 11). If the question includes a special requirement, "
        "such as outputting a dictionary, please fulfill that specific request. Otherwise, always output "
        "a list of strings, even if there is only one answer item.\n"
        "Please place your answers between the ``` and ```.\n\n"
        "### Notes:\n"
        "The table may include non-standard formats for numbers or dates.\n"
        "For numbers, formats may include Roman numerals, English words, or scientific notation. "
        "Whenever possible, please convert these into Arabic numerals in your response.\n"
        "Additionally, some columns are derived through mathematical operations. For example, `total_add_10` "
        "indicates that the values in this column are obtained by adding 10 to the original values. "
        "You should return the original values (i.e., subtract 10 from the current values). "
        "Furthermore, for any values ending with \"k\" or \"K\", convert them into standard Arabic numerals.\n"
        "Please convert all dates to the format %Y-%m-%d <other time elements> (in Arabic numerals) in your response.\n"
        "Additionally, even if there are duplicate answer items, you need to output all of them.\n\n"
        "# Example Output Format:\n"
        "```\n['answer_item_1', 'answer_item_2', ..., 'answer_item_n']\n```\n\n"
        "Please do not generate any text after outputting the final answer."
    )
    user_prompt = "table information:\n{}\nQuestion: {}\nAnswer:".format(table_infos, query)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def get_result(client, prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0,
        max_tokens=8192,
        timeout=36000,
    )
    logging.info(response)
    content = response.choices[0].message.content
    tokens = response.usage.prompt_tokens
    return content, tokens


def get_result_with_retries(client, prompt, model, max_retries=2, delay=1):
    for attempt in range(max_retries):
        try:
            return get_result(client, prompt, model)
        except Exception:
            logging.warning("Attempt {} failed:\n{}".format(attempt + 1, traceback.format_exc()))
            time.sleep(delay)
    return "", 0


def process_one(index, conv, client, tables_path, table_format, table_list, model, table_source, top_k):
    try:
        prompt = build_prompt(
            conv, tables_path, table_format, table_list, table_source, top_k
        )
        logging.info("Prompt [{}]: {}".format(index, prompt))
        output_str, use_tokens = get_result_with_retries(client, prompt, model)
        logging.info("Output [{}]: {}".format(index, output_str))
        precision, recall, f1 = cal_score3(conv["answer"], output_str)
        result = dict(conv)
        result.update(
            {
                "output_text": output_str,
                "tokens": use_tokens,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        return result
    except Exception:
        logging.error(traceback.format_exc())
        return None


def calculate_metrics(dt, elapsed_hours=0.0):
    if not dt:
        return {
            "average_precision": 0,
            "average_recall": 0,
            "average_f1": 0,
            "data_num": 0,
            "max_tokens": 0,
            "min_tokens": 0,
            "elapsed_hours": round(elapsed_hours, 2),
            "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    prec = [x["precision"] for x in dt]
    rec = [x["recall"] for x in dt]
    f1s = [x["f1"] for x in dt]
    toks = [x["tokens"] for x in dt]
    return {
        "average_precision": sum(prec) / len(prec),
        "average_recall": sum(rec) / len(rec),
        "average_f1": sum(f1s) / len(f1s),
        "data_num": len(dt),
        "max_tokens": max(toks),
        "min_tokens": min(toks),
        "elapsed_hours": round(elapsed_hours, 2),
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def classify_data(dt):
    by_type = defaultdict(list)
    for item in dt:
        by_type[item.get("question_type", "unknown")].append(item)
    return by_type


def evaluate_all(
    model_path,
    json_path,
    table_json_path,
    output_path,
    tables_path,
    table_format,
    key,
    url,
    num_workers,
    table_source="gold",
    top_k=1,
):
    start_time = time.time()
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "output.jsonl")

    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    with open(table_json_path, "r", encoding="utf-8") as f:
        table_list = json.load(f)

    existing_ids = set()
    done = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("output_text") not in (None, ""):
                    done.append(item)
                    if "question_id" in item:
                        existing_ids.add(item["question_id"])
        logging.info("Already done: {}".format(len(done)))

    all_data = [x for x in all_data if x.get("question_id") not in existing_ids]
    if not all_data:
        logging.info("Nothing to run.")
        dt = done
    else:
        client = OpenAI(api_key=key, base_url=url)
        logging.info("Running {} items with {} workers".format(len(all_data), num_workers))
        new_results = [None] * len(all_data)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_one,
                    idx,
                    conv,
                    client,
                    tables_path,
                    table_format,
                    table_list,
                    model_path,
                    table_source,
                    top_k,
                ): idx
                for idx, conv in enumerate(all_data)
            }
            with tqdm.tqdm(total=len(all_data), desc="TableQA") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    new_results[idx] = future.result()
                    pbar.update(1)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in done:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            for item in new_results:
                if item is not None:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(output_file, "r", encoding="utf-8") as f:
            dt = [json.loads(line) for line in f if line.strip()]

    elapsed_hours = (time.time() - start_time) / 3600.0

    with open(os.path.join(output_path, "output.json"), "w", encoding="utf-8") as f:
        json.dump(dt, f, ensure_ascii=False, indent=4)

    by_type = classify_data(dt)
    results = []
    for qtype, items in sorted(by_type.items()):
        r = calculate_metrics(items)
        r["classification"] = qtype
        results.append(r)
    r = calculate_metrics(dt, elapsed_hours=elapsed_hours)
    r["classification"] = "Overall"
    results.append(r)

    save_csv(results, os.path.join(output_path, "result.csv"))
    logging.info("Done. Results saved. elapsed_hours={:.2f}".format(elapsed_hours))


def main():
    parser = argparse.ArgumentParser(
        description="Direct TableQA evaluation on gold or retrieved tables"
    )
    parser.add_argument(
        "--json-path", type=str, default="data/query_data/TCRAG_Questions.json"
    )
    parser.add_argument("--tables-path", type=str, default="data/table/TCRAG_8k")
    parser.add_argument(
        "--table-json-path", type=str, default="data/query_data/TCRAG_Tables.json"
    )
    parser.add_argument("--output-path", type=str, default="results/QA_TableQA")
    parser.add_argument(
        "--table-format",
        type=str,
        default="markdown",
        choices=["markdown", "sql", "html", "latex", "xml", "json", "df", "csv", "mixed"],
        help="'mixed' uses each table's own table_format field from the table JSON.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Thinking-2507",
    )
    parser.add_argument("--key", type=str, default="xxx")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8004/v1")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel threads")
    parser.add_argument(
        "--table-source",
        type=str,
        default="gold",
        choices=["gold", "dense", "rerank"],
        help="Table source: gold (highlighted_table), dense (output_text top-k), "
        "or rerank (output_text_rerank top-k).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top retrieved tables when table-source is dense or rerank.",
    )
    args = parser.parse_args()

    setup_logging(args.output_path)
    logging.info(vars(args))

    evaluate_all(
        model_path=args.model_path,
        json_path=args.json_path,
        table_json_path=args.table_json_path,
        output_path=args.output_path,
        tables_path=args.tables_path,
        table_format=args.table_format,
        key=args.key,
        url=args.url,
        num_workers=args.num_workers,
        table_source=args.table_source,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
