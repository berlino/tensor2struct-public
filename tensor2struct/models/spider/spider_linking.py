import attr
import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tensor2struct.utils import batched_sequence
from tensor2struct.contexts import knowledge_graph
from tensor2struct.utils import registry, gumbel
from tensor2struct.modules import rat, lstm, embedders, energys

import logging

logger = logging.getLogger("tensor2struct")


def get_graph_from_relations(desc, relations2id):
    """
    Protocol: the graph is contructed based on four keys of desc:
    question, columns, tables
    **MIND THE ORDER OF SECTIONS**
    """
    sections = [("q", len(desc["question"]))]
    if "columns" in desc:
        sections.append(("col", len(desc["columns"])))
    if "tables" in desc:
        sections.append(("tab", len(desc["tables"])))

    relations = [desc["schema_relations"], desc["sc_relations"], desc["cv_relations"]]
    relation_graph = knowledge_graph.KnowledgeGraph(sections, relations2id)
    for relation in relations:
        relation_graph.add_relations_to_graph(relation)
    return relation_graph.get_relation_graph()


def get_schema_graph_from_relations(desc, relations2id):
    sections = []
    if "columns" in desc:
        sections.append(("col", len(desc["columns"])))
    if "tables" in desc:
        sections.append(("tab", len(desc["tables"])))
    relations = [desc["schema_relations"]]
    relation_graph = knowledge_graph.KnowledgeGraph(sections, relations2id)
    for relation in relations:
        relation_graph.add_relations_to_graph(relation)
    return relation_graph.get_relation_graph()


@attr.s
class RelationMap:
    q_len = attr.ib(default=None)
    c_len = attr.ib(default=None)
    t_len = attr.ib(default=None)

    predefined_relation = attr.ib(default=None)
    ct_relation = attr.ib(default=None)
    qq_relation = attr.ib(default=None)
    qc_relation = attr.ib(default=None)
    qt_relation = attr.ib(default=None)
    cq_relation = attr.ib(default=None)
    tq_relation = attr.ib(default=None)


@registry.register("schema_linking", "spider_string_matching")
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
            t_len=len(desc["tables"]),
            predefined_relation=relations_t,
        )
        return relation_obj


def argmax(logits, device, dim):
    max_id = torch.argmax(logits, dim=dim, keepdim=True)
    one_hot = torch.zeros_like(logits).to(device).scatter_(dim, max_id, 1)
    return one_hot


@registry.register("schema_linking", "bilinear_matching")
class BilinearLinking(nn.Module):
    def __init__(
        self,
        device,
        preproc,
        word_emb_size,
        num_latent_relations,
        hidden_size=300,
        recurrent_size=256,
        discrete_relation=True,
        norm_relation=True,
        symmetric_relation=False,
        combine_latent_relations=False,
        score_type="bilinear",
        learnable_embeddings=False,
        question_encoder=("shared-en-emb",),
        column_encoder=("shared-en-emb",),
        table_encoder=("shared-en-emb",),
    ):
        super().__init__()
        self.preproc = preproc
        self.vocab = preproc.vocab
        self.word_emb_size = word_emb_size
        self._device = device
        self.hidden_size = hidden_size
        self.discrete_relation = discrete_relation
        self.norm_relation = norm_relation
        self.num_latent_relations = num_latent_relations
        self.relations2id = preproc.relations2id
        self.recurrent_size = recurrent_size
        self.dropout = 0.0

        score_funcs = {
            "bilinear": lambda: energys.Bilinear(
                hidden_size, num_latent_relations, include_id=True
            ),
            "mlp": lambda: energys.MLP(hidden_size, num_latent_relations),
        }

        # build modules
        if learnable_embeddings:
            self.en_learnable_words = self.vocab
        else:
            self.en_learnable_words = None
        shared_modules = {
            "shared-en-emb": embedders.LookupEmbeddings(
                self._device,
                self.vocab,
                self.preproc.word_emb,
                self.word_emb_size,
                learnable_words=self.en_learnable_words,
            ),
        }

        if self.preproc.use_ch_vocab:
            self.ch_vocab = preproc.ch_vocab
            if learnable_embeddings:
                self.ch_learnable_words = self.ch_vocab
            else:
                self.ch_learnable_words = None
            shared_modules["shared-ch-emb"] = embedders.LookupEmbeddings(
                self._device,
                self.ch_vocab,
                self.preproc.ch_word_emb,
                self.preproc.ch_word_emb.dim,
                learnable_words=self.ch_learnable_words,
            )
            shared_modules["ch-bilstm"] = lstm.BiLSTM(
                input_size=self.preproc.ch_word_emb.dim,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                use_native=False,
                summarize=False,
            )
            shared_modules["ch-bilstm-native"] = lstm.BiLSTM(
                input_size=self.preproc.ch_word_emb.dim,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                use_native=True,
                summarize=False,
            )

        self.question_encoder = self._build_modules(
            question_encoder, shared_modules=shared_modules
        )
        self.column_encoder = self._build_modules(
            column_encoder, shared_modules=shared_modules
        )
        self.table_encoder = self._build_modules(
            table_encoder, shared_modules=shared_modules
        )

        self.combine_latent_relations = combine_latent_relations
        if combine_latent_relations:
            self.string_link = StringLinking(device, preproc)

        self.symmetric_relation = symmetric_relation
        assert self.symmetric_relation
        if self.symmetric_relation:
            relations = ("qc", "qt")
        else:
            relations = ("qc", "cq", "tq", "qt")
        self.relation_score_dic = nn.ModuleDict(
            {k: score_funcs[score_type]() for k in relations}
        )

        if discrete_relation:
            self.temperature = 1  # for gumbel

        if not norm_relation:  # then norm q/col/tab
            self.null_q_token = nn.Parameter(torch.zeros([1, hidden_size]))
            self.null_c_token = nn.Parameter(torch.zeros([1, hidden_size]))
            self.null_t_token = nn.Parameter(torch.zeros([1, hidden_size]))

    def _build_modules(self, module_types, shared_modules=None):
        module_builder = {
            "en-emb": lambda: embedders.LookupEmbeddings(
                self._device,
                self.vocab,
                self.preproc.word_emb,
                self.word_emb_size,
                learnable_words=self.en_learnable_words,
            ),
            "bilstm": lambda: lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False,
                use_native=False,
            ),
            "bilstm-native": lambda: lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False,
                use_native=True,
            ),
        }

        modules = []
        for module_type in module_types:
            if module_type in shared_modules:
                modules.append(shared_modules[module_type])
            else:
                modules.append(module_builder[module_type]())
        return torch.nn.Sequential(*modules)

    def compute_relation_score(self, x1, x2, boudaries, score_type):
        """
        x1, x2: len * relation_emb_size
        """
        x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)  # len * 1 * emb_size
        len_1, _, rs = x1.size()
        len_2, _, rs = x2.size()
        _x1 = x1.expand(len_1, len_2, rs)
        _x2 = x2.expand(len_2, len_1, rs).transpose(0, 1)
        relation_scores = self.relation_score_dic[score_type](_x1, _x2)

        # TODO: optimize this code
        res = []
        for s, e in zip(boudaries, boudaries[1:]):
            max_val, max_id = torch.max(relation_scores[:, s:e, :], dim=1, keepdim=True)
            res.append(max_val)
        res_v = torch.cat(res, dim=1)
        return res_v

    def normalize_relation_score(self, relation_scores):
        """
        relation_scores: either dim_1 or dim_2 will be normalized
        """
        if not self.norm_relation:
            norm_dim = 1
        else:
            norm_dim = 2

        if self.discrete_relation:
            device = relation_scores.device
            if self.training:
                r = gumbel.gumbel_softmax_sample(
                    relation_scores, self.temperature, device, norm_dim
                )
            else:
                r = argmax(relation_scores, device, norm_dim)
        else:
            r = torch.softmax(relation_scores, dim=norm_dim)
        return r

    def get_symmetric_relation(self, x1, x2, boudaries, score_type, ignore_null=True):
        x1_type, x2_type = score_type
        assert x1_type == "q"  # qc, qt
        # pack the null token
        if not self.norm_relation:
            null_token_1 = getattr(self, f"null_{x1_type}_token")
            x1 = torch.cat([x1, null_token_1], 0)
            null_token_2 = getattr(self, f"null_{x2_type}_token")
            x2 = torch.cat([x2, null_token_2], 0)
            boudaries.append(boudaries[-1] + 1)

        relation_scores = self.compute_relation_score(x1, x2, boudaries, score_type)
        r1 = self.normalize_relation_score(relation_scores)
        r2 = self.normalize_relation_score(relation_scores.transpose(0, 1))

        # unpack the null tokens
        if not self.norm_relation and ignore_null:
            r1 = r1[:-1, :-1, :]
            r2 = r2[:-1, :-1, :]
        return r1, r2

    def get_q_ct_relations(self, desc, ignore_null, column_type):
        q_enc, _ = self.question_encoder([[desc["question"]]])
        if column_type:
            c_enc, c_boudaries = self.column_encoder(
                [[col[1:] for col in desc["columns"]]]
            )
        else:
            c_enc, c_boudaries = self.column_encoder([desc["columns"]])

        t_enc, t_boudaries = self.table_encoder([desc["tables"]])

        q_enc, c_enc, t_enc = q_enc.select(0), c_enc.select(0), t_enc.select(0)
        c_boudaries, t_boudaries = c_boudaries[0].tolist(), t_boudaries[0].tolist()

        qc_relation, cq_relation = self.get_symmetric_relation(
            q_enc, c_enc, c_boudaries, "qc", ignore_null=ignore_null
        )
        qt_relation, tq_relation = self.get_symmetric_relation(
            q_enc, t_enc, t_boudaries, "qt", ignore_null=ignore_null
        )
        return qc_relation, cq_relation, qt_relation, tq_relation

    def forward_unbatched(self, desc, ignore_null=True, column_type=True):
        qc_relation, cq_relation, qt_relation, tq_relation = self.get_q_ct_relations(
            desc, ignore_null, column_type
        )

        ct_relation_np = get_schema_graph_from_relations(desc, self.relations2id)
        ct_relation = torch.LongTensor(ct_relation_np).to(self._device)

        if self.combine_latent_relations:
            r = self.string_link(desc)
            predefined_relation = r.predefined_relation
        else:
            predefined_relation = None

        relations = RelationMap(
            q_len=len(desc["question"]),
            c_len=len(desc["columns"]),
            t_len=len(desc["tables"]),
            predefined_relation=predefined_relation,
            qc_relation=qc_relation,
            cq_relation=cq_relation,
            qt_relation=qt_relation,
            tq_relation=tq_relation,
            ct_relation=ct_relation,
        )
        return relations

    def forward(self, desc, ignore_null=True, column_type=True):
        return self.forward_unbatched(
            desc, ignore_null=ignore_null, column_type=column_type
        )


@registry.register("schema_linking", "sinkhorn_matching")
class SinkhornLinking(BilinearLinking):
    def __init__(
        self,
        device,
        preproc,
        num_latent_relations,
        word_emb_size,
        recurrent_size=256,
        discrete_relation=True,
        norm_relation=True,
        symmetric_relation=False,
        combine_latent_relations=False,
        question_encoder=("shared-en-emb", "bilstm-native"),
        column_encoder=("shared-en-emb", "bilstm-native"),
        table_encoder=("shared-en-emb", "bilstm-native"),
    ):
        super().__init__(
            device=device,
            preproc=preproc,
            word_emb_size=word_emb_size,
            num_latent_relations=num_latent_relations,
            discrete_relation=False,
            norm_relation=False,
            symmetric_relation=True,
            hidden_size=recurrent_size,
            recurrent_size=recurrent_size,
            combine_latent_relations=combine_latent_relations,
            question_encoder=question_encoder,
            column_encoder=column_encoder,
            table_encoder=table_encoder,
            score_type="bilinear",
            learnable_embeddings=False,
        )
        self.sh_temperature = 0.8
        self.num_sh_it = 16

        self.null_q_token = nn.Parameter(torch.zeros([1, recurrent_size]))
        self.null_c_token = nn.Parameter(torch.zeros([1, recurrent_size]))
        self.null_t_token = nn.Parameter(torch.zeros([1, recurrent_size]))

    def normalize_relation_score(self, relation_scores):
        x_len, _x_len, num_r = relation_scores.size()
        assert x_len == _x_len
        it_scores = relation_scores
        for _ in range(self.num_sh_it):
            it_scores = it_scores - torch.logsumexp(it_scores, dim=1, keepdim=True)
            it_scores = it_scores - torch.logsumexp(it_scores, dim=0, keepdim=True)
        prob_m = torch.exp(it_scores)
        return prob_m

    def compute_relation_score(self, x1, x2, score_type):
        """
        x1, x2: len * relation_emb_size
        """
        x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)  # len * 1 * emb_size
        len_1, _, rs = x1.size()
        len_2, _, rs = x2.size()
        _x1 = x1.expand(len_1, len_2, rs)
        _x2 = x2.expand(len_2, len_1, rs).transpose(0, 1)
        relation_scores = self.relation_score_dic[score_type](_x1, _x2)
        return relation_scores

    def get_symmetric_relation(self, x1, x2, boudaries, score_type, ignore_null=True):
        q_type, ct_type = score_type
        assert q_type == "q"  # qc, qt
        q_len, feat_dim = x1.size()
        ct_len, feat_dim = x2.size()

        pad_len = max(q_len, ct_len) + 1
        null_token_q = getattr(self, f"null_{q_type}_token").expand(pad_len - q_len, -1)
        q_input = torch.cat([x1, null_token_q], 0)
        null_token_ct = getattr(self, f"null_{ct_type}_token").expand(
            pad_len - ct_len, -1
        )
        ct_input = torch.cat([x2, null_token_ct], 0)

        relation_scores = self.compute_relation_score(q_input, ct_input, score_type)
        q_ct_r = self.normalize_relation_score(relation_scores)

        # merge prob mass of ct
        m_q_ct_r_sum = []
        m_q_ct_r_avg = []
        for s, e in zip(boudaries, boudaries[1:]):
            sum_val = torch.sum(q_ct_r[:q_len, s:e, :], dim=1, keepdim=True)
            m_q_ct_r_sum.append(sum_val)
            avg_val = torch.mean(q_ct_r[:q_len, s:e, :], dim=1, keepdim=True)
            m_q_ct_r_avg.append(avg_val)
        m_q_ct_r = torch.cat(m_q_ct_r_sum, dim=1)

        m_q_ct_r_avg = torch.cat(m_q_ct_r_avg, dim=1)
        m_ct_q_r = m_q_ct_r_avg.transpose(0, 1)
        return m_q_ct_r, m_ct_q_r

    def merge_duplicates(self, items):
        # input: list of list of words
        new_item_list = []
        new_item2id = {}
        id_map = []
        for i, item in enumerate(items):
            t_item = tuple(item)
            if t_item not in new_item_list:
                new_item2id[t_item] = len(new_item_list)
                new_item_list.append(t_item)
            id_map.append(new_item2id[t_item])
        return new_item_list, id_map

    def get_q_ct_relations(self, desc, ignore_null, column_type):
        # create mapping
        q_enc, _ = self.question_encoder([[desc["question"]]])
        if column_type:
            raw_columns = [col[1:] for col in desc["columns"]]
        else:
            raw_columns = desc["columns"]
        new_columns, column_id_map = self.merge_duplicates(raw_columns)
        c_enc, c_boudaries = self.column_encoder([new_columns])
        new_tables, table_id_map = self.merge_duplicates(desc["tables"])
        t_enc, t_boudaries = self.table_encoder([new_tables])

        # compute relations
        q_enc, c_enc, t_enc = q_enc.select(0), c_enc.select(0), t_enc.select(0)
        c_boudaries, t_boudaries = c_boudaries[0].tolist(), t_boudaries[0].tolist()
        m_qc_relation, m_cq_relation = self.get_symmetric_relation(
            q_enc, c_enc, c_boudaries, "qc", ignore_null=ignore_null
        )
        m_qt_relation, m_tq_relation = self.get_symmetric_relation(
            q_enc, t_enc, t_boudaries, "qt", ignore_null=ignore_null
        )

        # map it back
        column_id_map = torch.LongTensor(column_id_map).to(self._device)
        table_id_map = torch.LongTensor(table_id_map).to(self._device)
        qc_relation = m_qc_relation.index_select(1, column_id_map)
        cq_relation = m_cq_relation.index_select(0, column_id_map)
        qt_relation = m_qt_relation.index_select(1, table_id_map)
        tq_relation = m_tq_relation.index_select(0, table_id_map)

        return qc_relation, cq_relation, qt_relation, tq_relation
