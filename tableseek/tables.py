"""Table corpus loading for TableSeek (text-only; no dense embeddings)."""

import json
import os
import re
import logging
import pandas as pd
from tqdm import tqdm

from tableseek.reformat import (
    df2mk, df2html, df2sql, df2latex, df2xml, df2json, df2csv,
    df2sentence, df2sentence_shuffled,
)


def build_table_key(db_id, table_path):
    return "{}::{}".format(db_id, table_path)


def trans_tables(table_name, df, table_format):
    format_map = {
        "markdown": lambda d: df2mk(d),
        "html": lambda d: df2html(d),
        "sql": lambda d: df2sql(d, table_name),
        "latex": lambda d: df2latex(d),
        "xml": lambda d: df2xml(d),
        "json": lambda d: df2json(d),
        "df": lambda d: str(d),
        "csv": lambda d: df2csv(d),
        "sentence": lambda d: df2sentence(d),
        "sentence_shuffled": lambda d: df2sentence_shuffled(d),
    }
    table_infos = format_map[table_format](df)
    table_infos = re.sub(r" +", " ", table_infos)
    return "{}:\n{}\n\n".format(table_name, table_infos)


def load_all_tables(tables_json_path, tables_path, table_format):
    """
    Load all tables into a dict keyed by ``db_id::table_path``.

    Each value is ``{"table_str": <serialized table text>}``.
    """
    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    table_store = {}

    for item in tqdm(tables_data, desc="Loading tables"):
        db_id = item["db_id"]
        table_path = item["table_path"]
        table_name = item["table_name"]
        table_format_used = (
            item["table_format"] if table_format == "default" else table_format
        )

        key = build_table_key(db_id, table_path)

        if "table_data" in item:
            table_data = item["table_data"]
            if isinstance(table_data, list) and len(table_data) >= 2:
                columns = table_data[0]
                data_rows = table_data[1:]
                df = pd.DataFrame(data_rows, columns=columns)
            else:
                df = pd.DataFrame()
        else:
            table_file = os.path.join(tables_path, db_id, "{}.csv".format(table_path))
            df = pd.read_csv(table_file, low_memory=False)

        table_str = trans_tables(table_name, df, table_format_used)
        table_store[key] = {"table_str": table_str}

    logging.info("Loaded %d tables.", len(table_store))
    return table_store
