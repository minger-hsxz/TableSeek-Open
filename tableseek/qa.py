import logging
import re
from tableseek.patterns import get_llm_client

def extract_text_after_tag(text, tag="</think>"):
    return text.split(tag)[-1].strip()

QA_SYSTEM_PROMPT = '''### Requirements:
Please read the following table and then answer the questions based on the table.  
Organize your answers into a list of strings, with each element being an answer item (The number of answer items is less than 11). If the question includes a special requirement, such as outputting a dictionary, please fulfill that specific request. Otherwise, always output a list of strings, even if there is only one answer item.  
Please place your answers between the ``` and ```.

### Notes:
The table may include non-standard formats for numbers or dates. 
For numbers, formats may include Roman numerals, English words, or scientific notation. Whenever possible, please convert these into Arabic numerals in your response.
Additionally, some columns are derived through mathematical operations. For example, `total_add_10` (other operations or variants may also exist) indicates that the values in this column are obtained by adding 10 to the original values. You should return the original values (i.e., subtract 10 from the current values). Furthermore, for any values ending with "k" or "K", convert them into standard Arabic numerals.
Please convert all dates to the format %Y-%m-%d <other time elements> (in Arabic numerals) in your response.

Additionally, even if there are duplicate answer items, you need to output all of them.

# Example Output Format:  
```
['answer_item_1', 'answer_item_2', ..., 'answer_item_n']
```

Please do not generate any text after outputting the final answer.'''

TABFACT_SYSTEM_PROMPT = (
    "Determine whether the given statement is supported by the retrieved table.\n\n"
    "Requirements:\n"
    "- Use only information from the table.\n"
    "- If the statement is fully supported by the table, output 'true'.\n"
    "- If the statement is contradicted by the table, output 'false'.\n"
    "- If the retrieved table does not contain the columns or entities needed to verify the statement "
    "(i.e., schema mismatch), output 'mismatch'.\n"
    "- Do not provide any explanation.\n"
    "- Output only one word: 'true', 'false', or 'mismatch'.\n"
)

def build_table_context(retrieved_keys: list, table_store: dict) -> str:
    parts = []
    for key in retrieved_keys:
        table_str = table_store.get(key, {}).get('table_str', '')
        if table_str:
            parts.append(f"[Table: {key}]\n{table_str}")
    return "\n\n".join(parts) if parts else "(no table retrieved)"


def build_tabfact_messages(statement: str, table_infos_str: str) -> list:
    user_prompt = (
        f"table information:\n{table_infos_str}\n\n"
        f"Statement: {statement}\nAnswer:"
    )
    return [
        {"role": "system", "content": TABFACT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]


def build_qa_messages(query: str, table_infos_str: str) -> list:
    user_prompt = (
        f"table information:\n{table_infos_str}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]


def llm_qa(
    query: str,
    table_infos_str: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    max_retries: int = 3,
) -> str:
    """Call QA LLM. Returns raw output string."""
    client = get_llm_client(llm_base_url, llm_api_key)
    messages = build_qa_messages(query, table_infos_str)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=8192,
                timeout=18000,
                top_p=1.0,
                seed=42,
            )
            raw_output = response.choices[0].message.content or ""
            logging.info(f"LLM QA raw output for {query} and table {table_infos_str}: {raw_output}")
            return raw_output
        except Exception as e:
            last_error = e
            logging.warning(f"QA LLM attempt {attempt}/{max_retries} failed: {e}")

    logging.error(f"All {max_retries} QA LLM attempts failed for [{query[:60]}]: {last_error}")
    return ""


def is_empty_output(raw_output: str) -> bool:
    """Heuristic: consider QA output empty if it contains no list-like content."""
    stripped = extract_text_after_tag(raw_output)
    if not stripped:
        return True
    # Look for the answer block between backticks
    match = re.search(r'```(.*?)```', stripped, re.DOTALL)
    if not match:
        return True
    content = match.group(1).strip()
    # Empty list or whitespace only
    if content in ('[]', '') or not content:
        return True
    return False

def llm_tabfact(
    statement: str,
    table_infos_str: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    max_retries: int = 3,
) -> str:
    """Call tabfact LLM. Returns raw output string."""
    client = get_llm_client(llm_base_url, llm_api_key)
    messages = build_tabfact_messages(statement, table_infos_str)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=8192,
                timeout=18000,
                top_p=1.0,
                seed=42,
            )
            # return response.choices[0].message.content or ""
            raw_output = response.choices[0].message.content or ""
            logging.info(f"LLM QA raw output for {statement} and table {table_infos_str}: {raw_output}")
            return raw_output
        except Exception as e:
            last_error = e
            logging.warning(f"Tabfact LLM attempt {attempt}/{max_retries} failed: {e}")

    logging.error(f"All {max_retries} tabfact LLM attempts failed for [{statement[:60]}]: {last_error}")
    return ""

def is_tabfact_mismatch(raw_output: str) -> bool:
    """Heuristic: consider tabfact output a schema mismatch if no true/false signal is present."""
    stripped = extract_text_after_tag(raw_output).strip().lower()
    if not stripped:
        return True
    pred_clean = re.sub(r"[^\w\s]", " ", stripped)
    pred_clean = re.sub(r"\s+", " ", pred_clean).strip()
    has_true = bool(re.search(r"\btrue\b", pred_clean))
    has_false = bool(re.search(r"\bfalse\b", pred_clean))
    has_mismatch = bool(re.search(r"\bmismatch\b", pred_clean))
    if has_mismatch:
        return True
    if has_true or has_false:
        return False
    return True

def finalize_tabfact_output(raw_output: str) -> str:
    """If the final tabfact result is still mismatch, treat it as false."""
    if is_tabfact_mismatch(raw_output):
        return "false"
    return raw_output
