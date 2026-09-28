import pandas as pd
import io
import numpy as np
from random import seed, sample
import random
from bs4 import BeautifulSoup
import json
import re
import sqlparse
import traceback
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
from xml.dom import minidom
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

def df2sentence(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a sentence representation.
    For each row, generate sentences like:
    Col1 is val1. Col2 is val2. ...
    and concatenate them into a single string.

    Args:
        df (pd.DataFrame): Input table

    Returns:
        str: Complete string of all row sentences concatenated
    """
    sentences = []
    for _, row in df.iterrows():
        parts = [f"{col} is {row[col]}" for col in df.columns]
        sentence = ". ".join(parts) + "."
        sentences.append(sentence)

    # Join all sentences with newlines
    return "\n".join(sentences)


def df2sentence_shuffled(df: pd.DataFrame, seed: int = 42) -> str:
    """
    Convert a pandas DataFrame to a sentence representation.
    For each row, randomly shuffle the column order and generate sentences like:
    Col_i is val_i. Col_j is val_j. ...
    and concatenate them into a single string.

    Args:
        df (pd.DataFrame): Input table
        seed (int): Random seed for reproducibility, default 42

    Returns:
        str: Complete string of all row sentences concatenated
    """
    sentences = []
    rng = random.Random(seed)

    for _, row in df.iterrows():
        items = list(row.items())
        rng.shuffle(items)  # Shuffle once per row, order naturally differs
        parts = [f"{col} is {val}" for col, val in items]
        sentence = ". ".join(parts) + "."
        sentences.append(sentence)

    # Join all sentences with newlines
    return "\n".join(sentences)

def df2csv(df):
    """
    Convert a pandas DataFrame to a CSV format string.

    Args:
        df (pd.DataFrame): The pandas DataFrame to convert.

    Returns:
        str: CSV format string.
    """
    # Use StringIO to capture CSV output
    output = io.StringIO()

    # Write DataFrame to StringIO object
    df.to_csv(output, index=False)

    # Get the CSV string
    csv_string = output.getvalue()

    # Close the StringIO object
    output.close()

    return csv_string

def is_number(s):
    # Regex to match integer or decimal
    pattern = re.compile(r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$')
    return bool(pattern.match(str(s).strip()))


def format_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Format all floats in the DataFrame to a consistent precision for comparison."""

    def format_float(x):
        if isinstance(x, float):
            return f"{x:.10e}"
        return x

    return df.applymap(format_float)

def df2mk(df):
    output = io.StringIO()
    df.to_markdown(buf=output, index=False)
    return output.getvalue()

def mk2df(original_table):
    seed(42)

    # Read the original table
    original_df = pd.read_csv(io.StringIO(original_table), sep='|', skipinitialspace=True)
    original_df.columns = original_df.columns.str.strip()
    original_df = original_df.dropna(axis=1, how='all')
    # Remove separator row like |-----:|------:|------------:|...
    original_df = original_df.drop(index=0)
    original_df = original_df.applymap(lambda x: float(x) if is_number(x) else str(x).strip())
    original_df.columns = original_df.columns.str.strip()
    return original_df

def df2html(df: pd.DataFrame) -> str:
    """Convert a DataFrame to an HTML table string."""
    # Convert datetime columns to string
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df.to_html(index=False)

def html2df(html_str: str) -> pd.DataFrame:
    """Convert an HTML table string to a DataFrame and preserve original whitespace."""
    soup = BeautifulSoup(html_str, 'html.parser')
    table = soup.find('table')

    # Extract headers
    headers = [th.get_text(separator=" ", strip=False) for th in table.find_all('th')]

    # Extract rows
    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip the header row
        cells = [td.get_text(separator=" ", strip=False) for td in tr.find_all('td')]
        rows.append(cells)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df.where(pd.notnull(df), None)

def df2latex(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a LaTeX table string without caption and label."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    num_columns = len(df.columns)
    column_format = '|' + '|'.join(['l'] * num_columns) + '|'  # Add vertical lines between columns

    latex_str = "\\begin{table}[!ht]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\begin{{tabular}}{{{column_format}}}\n"
    latex_str += "\\hline\n"

    header = ' & '.join([str(col) for col in df.columns]) + ' \\\\\n'
    latex_str += header
    latex_str += "\\hline\n"

    for _, row in df.iterrows():
        row_str = ' & '.join([str(value) for value in row]) + ' \\\\\n'
        latex_str += row_str
        latex_str += "\\hline\n"

    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"

    return latex_str


def latex2df(latex_str: str) -> pd.DataFrame:
    """Convert a LaTeX table string to a DataFrame."""
    # Use regex to match table content
    match = re.search(r'\\begin{tabular}.*?\\end{tabular}', latex_str, re.DOTALL)
    if not match:
        raise ValueError("No tabular environment found in the LaTeX string.")

    tabular_content = match.group(0)

    # Remove LaTeX commands, keep only table data
    data_lines = re.sub(r'\\[a-zA-Z]*\{.*?\}', '', tabular_content)
    data_lines = re.sub(r'\\hline', '', data_lines)
    # data_lines = re.sub(r'\$.*?\$', '', data_lines)  # Intended to remove math formulas, but cells like '$123-$456' would be incorrectly stripped, so commented out
    # data_lines = re.sub(r'&', ' ', data_lines)
    data_lines = re.sub(r'\\\\', '\n', data_lines)

    if re.search(r'\{.*?\}', data_lines):
        # Exclude cases where braces in cells are easily confused with LaTeX syntax
        bracket_exclusions = ['Unplugged', 'L H', 'C1 C3', '0', '1', '3', '6', '7', '8', '9', 'Florida', 'punches',
                              'Nicolas Chouity', 'tot']
        bracket_exclusions_regex = '|'.join(re.escape(ex) for ex in bracket_exclusions)
        bracket_pattern = r'\{(?!' + bracket_exclusions_regex + r'\b).*?\}'
        data_lines = re.sub(bracket_pattern, '', data_lines)

    data_lines = data_lines.strip()

    # Set column headers
    data_line_list = data_lines.split('\n\n\n')
    column_names = re.split(r'\s+&\s+', data_line_list[0].strip())

    data = []
    for data_line in data_line_list[1:]:
        data.append(re.split(r'\s+&\s+', data_line.strip()))

    try:
        df = pd.DataFrame(data, columns=column_names)
        df = df.applymap(lambda x: float(x) if is_number(x) else str(x).strip())
        #df = format_floats(df)
    except Exception as e:
        print(f"data_line_list[-2]: {data_line_list[-2]}\ndata: {data}\ncolumn_names: {column_names}\n")
        raise e
    return df

def df2sql(df: pd.DataFrame, table_name: str) -> str:
    """Convert a DataFrame to an SQL create and insert statements string."""
    # SQL handling: support single quote in table strings, e.g. Alice's Dog
    columns = ', '.join(df.columns)
    create_table = f"CREATE TABLE {table_name} ({columns});\n"
    insert_into = f"INSERT INTO {table_name} ({columns}) VALUES\n"
    values = ',\n'.join(['(' + ', '.join(f"""'{str(value).replace("'", "''")}'""" for value in row) + ')' for row in df.values])
    return create_table + insert_into + values + ';'

def sql2df(sql_str: str) -> pd.DataFrame:
    """Convert an SQL insert statements string to a DataFrame."""
    statements = sqlparse.split(sql_str)
    insert_statement = next(stmt for stmt in statements if stmt.strip().upper().startswith('INSERT'))
    create_statement = next(stmt for stmt in statements if stmt.strip().upper().startswith('CREATE'))

    # Extract column names from CREATE TABLE statement
    create_tokens = sqlparse.parse(create_statement)[0].tokens
    column_names = []
    read_columns = False
    for token in create_tokens:
        if token.ttype is None and '(' in token.value:
            read_columns = True
        else:
            read_columns = False
        if read_columns and token.ttype is None:
            col_list = token.value[1:-1].split(',')
            for col in col_list:
                col_name = col.strip().split(' ')[0]
                if col_name and col_name not in ['(', ')']:
                    column_names.append(col_name)

    # Extract values from INSERT INTO statement
    values_str = insert_statement.split('VALUES', 1)[1].strip().strip(';')
    rows = re.findall(r'\((.*?)\)', values_str)

    # Clean up and create the DataFrame
    processed_rows = []
    for row in rows:
        values = [val.strip().strip("'").replace("''", "'") for val in row.split(',')]
        processed_rows.append(values)

    df = pd.DataFrame(processed_rows, columns=column_names)

    return df

def df2json(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a JSON table string with original float precision."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    json_str = df.to_json(orient='table', double_precision=15)

    json_data = json.loads(json_str)
    json_table = json.dumps(json_data, ensure_ascii=False, indent=4)

    return json_table


def json2df(json_str: str) -> pd.DataFrame:
    """Convert a JSON table string to a DataFrame with original float precision."""
    json_data = json.loads(json_str)

    col_names = [item["name"] for item in json_data["schema"]["fields"][1:]]
    data = [list(item.values())[1:] for item in json_data["data"]]

    df = pd.DataFrame(data, columns=col_names)

    df = df.applymap(lambda x: float(x) if is_number(x) else str(x).strip())
    #df = format_floats(df)
    return df

def adjust_xml_text(text: str) -> str:
    """Escape special characters in XML text."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;').replace("²", "2").replace("³", "3").replace("º", "o").replace("ʲ", "j").replace("ʷ", "w").replace("¹", "1")

def adjust_xml_attr(name: str) -> str:
    """Escape special characters in XML attribute names."""
    return name.replace(' ', '_').replace('&', '_').replace('<', '_').replace('>', '_').replace('"', '_').replace("'", '_').replace("²", "2").replace("³", "3").replace("º", "o").replace("ʲ", "j").replace("ʷ", "w").replace("¹", "1").replace('အင_သ_', '').replace('ရ_သ_', '').replace('ရ_သ_ခ_င_', '').replace('ခ_', '').replace('င_', '').replace('%', '_pppersent_').replace('(', '_lleft_brace_').replace(')', '_rright_brace_').replace('+', '_aadd_').replace('/', '_ddiv_').replace('?', '_question_').replace('#','_hhashtag_').replace(":","_ccolon_")

def clean_invalid_xml_chars(xml_str):
    # Corrected regex using Unicode character class range
    cleaned_xml = re.sub(r'[^\x09\x0A\x0D\x20-\xD7FF\uE000-\uFFFD\u10000-\u10FFFF]', '', xml_str)
    return cleaned_xml

def sanitize_xml_tags_in_string(input_string):
    # Regex to match all tags (including closing tags)
    def sanitize_tag(match):
        tag = match.group(1)  # Get tag name
        # If tag starts with digit, add underscore
        if tag[0].isdigit():
            tag = f"_{tag}"
        # Check if it's a closing tag, preserve closing tag form
        if match.group(0).startswith('</'):
            return f"</{tag}>"
        else:
            return f"<{tag}>"

    # Regex replace all tags (both opening and closing)
    sanitized_string = re.sub(r'</?([^>]+)>', sanitize_tag, input_string)
    return sanitized_string

def sanitize_xml_tags_in_string2(input_string):
    # Regex to match all tags
    def sanitize_tag(match):
        tag = match.group(1)  # Get tag name
        # If tag starts with digit, add underscore
        if tag[0].isdigit():
            tag = f"_{tag}"
        return f"<{tag}>"

    # Regex replace all tags
    sanitized_string = re.sub(r'<([^>]+)>', sanitize_tag, input_string)
    return sanitized_string

def adjust_xml_str(text: bytes) -> str:
    """Escape special characters in XML text."""
    return text.decode('utf-8').replace(b'\xc2\xb5'.decode('utf-8'), 'μ').replace(b'\xce\xb7'.decode('utf-8'), 'η')

def df2xml(df: pd.DataFrame, root_name='root', row_name='row') -> str:
    """Convert DataFrame to XML string with proper escaping."""
    root = Element(root_name)
    for i, row in df.iterrows():
        row_element = SubElement(root, row_name)
        for col in df.columns:
            col_name = adjust_xml_attr(col)
            cell = SubElement(row_element, col_name)
            #cell_value = str(row[col])
            cell_text = adjust_xml_text(str(row[col]))
            cell.text = cell_text
    # tostring has its own escaping; pre-processing would double-escape (e.g. Alice's dog -> Alice&amp;apos;s)
    xml_str_raw = tostring(root, encoding='utf-8')
    #xml_str = escape(xml_str)
    xml_str = adjust_xml_str(xml_str_raw).replace("nan", "null")
    xml_str = clean_invalid_xml_chars(xml_str)
    xml_str = sanitize_xml_tags_in_string(xml_str)
    try:
        xml_final_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    except Exception as e:
        print("new error")
        print(f"xml_str: {xml_str}\ndf.columns: {df.columns}\n")
        raise e
        exit()
    return xml_final_str

def xml2df(xml_str: str, root_name='root', row_name='row') -> pd.DataFrame:
    """Convert an xml table string to a DataFrame."""
    # Parse the XML string
    # xml_str = xml_str.replace('pppersent', '%')  # Handle % in column names
    root = ET.fromstring(xml_str)

    # Initialize a list to hold the rows of data
    data = []

    # Iterate over each row element in the XML
    for row_element in root.findall(row_name):
        row_data = {}
        for cell_element in row_element:
            col_name = cell_element.tag
            col_name = col_name.replace('pppersent', '%')  # Handle % in column names
            cell_value = cell_element.text
            if cell_value.isdigit():
                cell_value = int(cell_value)
            elif cell_value.replace('.', '', 1).isdigit():
                cell_value = float(cell_value)
            row_data[col_name] = cell_value
        data.append(row_data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    df = df.applymap(lambda x: float(x) if is_number(x) else str(x).strip())
    #df = format_floats(df)

    return df

if __name__ == '__main__':
    json_path= '../1018/generated_data_1018_spider.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    for data in all_data:
        try:
            small_content=mk2df(data['small_content'])
            new_content=mk2df(data['new_content'])
            shuffle_content=mk2df(data['shuffle_content'])
            new_shuffle_content=mk2df(data['new_shuffle_content'])
            c_html=df2html(new_content)
            d_html=html2df(c_html)
            #print(c_html)
            #print(d_html)
            c_latex = df2latex(new_content)
            d_latex = latex2df(c_latex)
            #print(c_latex)
            #print(d_latex)
            c_sql = df2sql(new_content,data['table_name'])
            d_sql = sql2df(c_sql)
            #print(c_sql)
            #print(d_sql)
            c_json = df2json(new_content)
            #print(c_json)
            d_json = json2df(c_json)
            c_xml = df2xml(new_content)
            #print(c_xml)
            d_xml = xml2df(c_xml)
            #print(d_xml)
            #exit()
        except Exception as e:
            error_message = traceback.format_exc()
            print(f"An error occurred: {error_message}")
            print("ori_context:")
            print(new_content)
            exit()