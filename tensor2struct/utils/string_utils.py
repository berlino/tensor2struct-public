import json
import re
from unidecode import unidecode


def load_jsonl(filename):
    examples = []
    with open(filename) as f:
        for line in f:
            _example = json.loads(line)
            examples.append(_example)
    return examples


def load_jsonl_table(filename):
    tables = dict()
    with open(filename) as f:
        for line in f:
            _table = json.loads(line)
            tables[_table["id"]] = _table
    return tables


def normalize_string(string: str) -> str:
    """
    These are the transformation rules used to normalize cell in column names in Sempre.  See
    ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
    ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
    rules here to normalize and canonicalize cells and columns in the same way so that we can
    match them against constants in logical forms appropriately.
    """
    # Normalization rules from Sempre
    # \u201A -> ,
    string = unidecode(string.lower())
    string = re.sub("‚", ",", string)
    string = re.sub("„", ",,", string)
    string = re.sub("[·・]", ".", string)
    string = re.sub("…", "...", string)
    string = re.sub("ˆ", "^", string)
    string = re.sub("˜", "~", string)
    string = re.sub("‹", "<", string)
    string = re.sub("›", ">", string)
    string = re.sub("[‘’´`]", "'", string)
    string = re.sub("[“”«»]", '"', string)
    string = re.sub("[•†‡²³]", "", string)
    string = re.sub("[‐‑–—−]", "-", string)
    string = re.sub("[ðø′″€⁄ªΣ]", "_", string)
    string = re.sub("[\\u0180-\\u0210]", "", string).strip()
    string = re.sub("[\\u0220-\\uFFFF]", "", string).strip()
    string = string.replace("\\n", "_")
    string = re.sub("\\s+", " ", string)
    # Canonicalization rules from Sempre.
    string = re.sub("[^\\w]", "_", string)
    string = re.sub("_+", "_", string)
    string = re.sub("_$", "", string)
    return string.strip("_")
