import re
import attr
import numpy as np
from typing import Dict, List, Union


@attr.s
class Section:
    name = attr.ib()
    num_items = attr.ib(factory=list)


class KnowledgeGraph:
    def __init__(self, sections: List[Section], relations2id: Dict):
        if not isinstance(sections[0], Section):  # list
            assert isinstance(sections[0], list) or isinstance(sections[0], tuple)
            new_sections = []
            for section in sections:
                new_section = Section(name=section[0], num_items=section[1])
                new_sections.append(new_section)
        sections = new_sections
        self.sections = sections

        self.relations2id = relations2id
        self.item_num = sum(section.num_items for section in sections)

        self.section_ind2graph_ind = {}
        self.graph_ind2section_ind = {}
        base = 0
        for section in sections:
            for i in range(section.num_items):
                self.section_ind2graph_ind[(section.name, i)] = base
                self.graph_ind2section_ind[base] = (section.name, i)
                base += 1

        self.graph = np.full((self.item_num, self.item_num), -1, dtype=np.int64)

    def add_relations_to_graph(self, link_dic: Dict):
        """
        Relation name should follow the template: 
        section1-section2:relation_name, like question-column:ColumnExactMatch
        """
        for link_name in link_dic:
            _match = re.match(r"(.*)-(.*):(.*)", link_name)
            if _match is None:
                raise Exception(f"Relation name {link_name} format is wrong")
            section1, section2, relation_name = _match.groups()
            assert link_name in self.relations2id

            for item_pair in link_dic[link_name]:
                item1_ind, item2_ind = item_pair
                graph_ind1 = self.section_ind2graph_ind[(section1, item1_ind)]
                graph_ind2 = self.section_ind2graph_ind[(section2, item2_ind)]
                self.graph[graph_ind1, graph_ind2] = self.relations2id[link_name]

    def _fill_graph_with_defaults(self):
        row_num, col_num = self.graph.shape
        for i in range(row_num):
            for j in range(col_num):
                if self.graph[i, j] == -1:
                    s1_name, _ = self.graph_ind2section_ind[i]
                    s2_name, _ = self.graph_ind2section_ind[j]
                    default_r = f"{s1_name}:{s2_name}-default"
                    self.graph[i, j] = self.relations2id[default_r]

    def get_relation_graph(self):
        self._fill_graph_with_defaults()
        return self.graph
