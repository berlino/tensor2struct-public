import attr
import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tensor2struct.utils import batched_sequence
from tensor2struct.contexts import knowledge_graph
from tensor2struct.utils import registry
from tensor2struct.modules import rat, lstm, embedders


def get_graph_from_relations(desc, relations2id):
    """
    Protocol: the graph is contructed based on four keys of desc:
    question, columns, tables, values
    **MIND THE ORDER OF SECTIONS**
    """
    sections = [("q", len(desc["question"]))]
    if "columns" in desc:
        sections.append(("col", len(desc["columns"])))
    if "values" in desc:
        sections.append(("val", len(desc["values"])))

    relations = [desc["schema_relations"], desc["sc_relations"], desc["cv_relations"]]
    relation_graph = knowledge_graph.KnowledgeGraph(sections, relations2id)
    for relation in relations:
        relation_graph.add_relations_to_graph(relation)
    return relation_graph.get_relation_graph()


def get_schema_graph_from_relations(desc, relations2id):
    sections = []
    if "columns" in desc:
        sections.append(("col", len(desc["columns"])))
    if "values" in desc:
        sections.append(("val", len(desc["values"])))
    relations = [desc["schema_relations"]]
    relation_graph = knowledge_graph.KnowledgeGraph(sections, relations2id)
    for relation in relations:
        relation_graph.add_relations_to_graph(relation)
    return relation_graph.get_relation_graph()


@attr.s
class RelationMap:
    q_len = attr.ib(default=None)
    c_len = attr.ib(default=None)
    v_len = attr.ib(default=None)

    predefined_relation = attr.ib(default=None)


@registry.register("schema_linking", "overnight_string_matching")
class StringLinking:
    def __init__(self, device, preproc):
        self._device = device
        self.relations2id = preproc.relations2id

    def __call__(self, desc):
        return self.link_one_example(desc)

    def link_one_example(self, desc):
        relation_np = get_graph_from_relations(desc, self.relations2id)
        relations_t = torch.LongTensor(relation_np).to(self._device)
        relation_obj = RelationMap(
            q_len=len(desc["question"]),
            c_len=len(desc["columns"]),
            v_len=len(desc["values"]),
            predefined_relation=relations_t,
        )
        return relation_obj
