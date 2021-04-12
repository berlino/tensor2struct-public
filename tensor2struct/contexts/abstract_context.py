import attr
import abc
import nltk
import string
import sqlite3
import timeout_decorator
from typing import Dict, List, Union
from tensor2struct.utils import string_utils

# fmt: off
_STOP_WORDS = {"", "", "all", "being", "-", "over", "through", "yourselves", "its", "before",
              "hadn", "with", "had", ",", "should", "to", "only", "under", "ours", "has", "ought", "do",
              "them", "his", "than", "very", "cannot", "they", "not", "during", "yourself", "him",
              "nor", "did", "didn", "'ve", "this", "she", "each", "where", "because", "doing", "some", "we", "are",
              "further", "ourselves", "out", "what", "for", "weren", "does", "above", "between", "mustn", "?",
              "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn",
              "of", "could", "against", "isn", "or", "own", "into", "while", "whom", "down", "wasn", "your",
              "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves",
              ":", "was", "until", "more", "himself", "on", "but", "don", "herself", "haven", "those", "he",
              "me", "myself", "these", "up", ";", "below", "'re", "can", "theirs", "my", "and", "would", "then",
              "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "in", "any", "if", "!",
              "again", "'ll", "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan",
              "'t", "'s", "our", "after", "most", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"}
STOP_WORDS = set(nltk.corpus.stopwords.words("english")).union(_STOP_WORDS)
PUNKS = set(a for a in string.punctuation)
# fmt: on


CellValueType = Union[str, float]
TableType = List[Dict[str, CellValueType]]


@attr.s
class Entity:
    start = attr.ib()
    end = attr.ib()
    kind = attr.ib()  # number, string
    norm_value = attr.ib()


class AbstractContext(metaclass=abc.ABCMeta):
    """
    Context of table-like environments
    TODO: integrate spacy for matching functionalities like stop words, number recognition
    """

    _context_cache = {}  # caching processed context

    @abc.abstractmethod
    def compute_schema_relations(self):
        pass

    @abc.abstractmethod
    def compute_schema_linking(self, tokens: List) -> Dict:
        pass

    @abc.abstractmethod
    def compute_cell_value_linking(self, tokens: List) -> Dict:
        pass

    @staticmethod
    def partial_match(x_list, y_list):
        if len(x_list) == 1 and x_list[0] in STOP_WORDS:
            return False
        if len(x_list) == 1 and len(y_list) == 1:
            return False
        # x_str = "_".join([string_utils.normalize_string(x) for x in x_list])
        # y_str = "_".join([string_utils.normalize_string(x) for x in y_list])
        x_str = " " + " ".join(x_list) + " "
        y_str = " " + " ".join(y_list) + " " 
        if x_str in y_str and x_str != y_str:
            return True
        else:
            return False

    @staticmethod
    def exact_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str == y_str:
            return True
        else:
            return False

    @staticmethod
    @timeout_decorator.timeout(15)
    def db_word_match(word, column, table, db_path):
        """
        The order of decoder matters as staticmethod returns description not functions
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or {column} like '% {word} %'  or {column} like '{word}'"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except Exception as e:
            return False

    @staticmethod
    def isnumber(word):
        text = word.replace(",", "").lower()
        try:
            number = float(text)
            return True, number
        except ValueError:
            return False, None

    @staticmethod
    def isstopword(word):
        return word in STOP_WORDS or word in PUNKS
