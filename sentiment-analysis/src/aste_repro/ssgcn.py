"""Paper-informed SSGCN extension using PyABSA's ASTE batch contract."""

from __future__ import annotations


def build_ssgcn_class():
    """Create the PyTorch class lazily, after the pinned runtime is installed."""
    import math

    import torch
    from torch import nn
    from torch.nn import functional as F
    from pyabsa.tasks.AspectSentimentTripletExtraction.models.model import EMCGCN

    class SSGCN(EMCGCN):
        """SSGCN channels adapted to the ASTEDataset edge-feature representation.

        The learned edge channel follows SSGCN's syntax/semantics interaction.
        PyABSA's four existing structural feature embeddings supply relative
        distance, dependency relation, POS-pair, and tree-distance channels.
        """

        def __init__(self, config):
            super().__init__(config)
            if self.num_layers != 1:
                raise ValueError("the PyABSA edge-refinement contract supports one GCN layer")
            del self.triplet_biaffine
            hidden_dim = int(config.hidden_dim)
            graph_dim = int(config.gcn_dim)
            self.attention_heads = int(config.get("ssgcn_attention_heads", 12))
            if graph_dim % self.attention_heads:
                raise ValueError("GCN width must be divisible by SSGCN attention heads")
            self.ssgcn_query = nn.Linear(hidden_dim, graph_dim)
            self.ssgcn_key = nn.Linear(hidden_dim, graph_dim)
            self.ssgcn_aspect_query = nn.Linear(hidden_dim, graph_dim)
            self.ssgcn_opinion_key = nn.Linear(hidden_dim, graph_dim)
            self.ssgcn_affine = nn.Linear(graph_dim, graph_dim)
            self.ssgcn_edge_classifier = nn.Linear(1, int(config.output_dim))

        def _tree_distances(self, distance_ids):
            vocab = getattr(self.config, "syn_post_vocab", None)
            if vocab is None or not hasattr(vocab, "itos"):
                raise ValueError("PyABSA tree-distance vocabulary is unavailable")
            ids_to_dist = {
                index: int(value)
                for index, value in enumerate(vocab.itos)
                if isinstance(value, int)
            }
            distances = torch.full_like(distance_ids, 4)
            for index, distance in ids_to_dist.items():
                distances.masked_fill_(distance_ids == index, min(max(distance, 0), 4))
            return distances

        def forward(self, inputs):
            token_ids = inputs["token_ids"]
            masks = inputs["masks"]
            word_pair_position = inputs["word_pair_position"]
            word_pair_deprel = inputs["word_pair_deprel"]
            word_pair_pos = inputs["word_pair_pos"]
            word_pair_synpost = inputs["word_pair_synpost"]

            hidden = self.dropout_output(self.bert(token_ids, masks)["last_hidden_state"])
            batch, seq_len = masks.shape
            pair_mask = masks[:, :, None].bool() & masks[:, None, :].bool()
            pair_mask_4d = pair_mask.unsqueeze(-1).to(hidden.dtype)

            relative_logits = self.post_emb(word_pair_position)
            dependency_logits = self.deprel_emb(word_pair_deprel)
            pos_pair_logits = self.postag_emb(word_pair_pos)
            tree_distance_logits = self.synpost_emb(word_pair_synpost)

            # Build the symmetric dependency probability matrix from PyABSA's
            # encoded dependency-arc feature and normalize each source row.
            # ID 0 is padding and ID 1 is PyABSA's unknown relation.
            adjacency = (word_pair_deprel > 1).to(hidden.dtype)
            adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
            adjacency = adjacency * pair_mask.to(hidden.dtype)
            syntax_attention = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)

            def split_heads(values):
                return values.view(
                    batch, seq_len, self.attention_heads, -1
                ).transpose(1, 2)

            queries = split_heads(self.ssgcn_query(hidden))
            keys = split_heads(self.ssgcn_key(hidden)).transpose(-2, -1)
            aspect_queries = split_heads(self.ssgcn_aspect_query(hidden))
            opinion_keys = split_heads(self.ssgcn_opinion_key(hidden)).transpose(-2, -1)
            scale = math.sqrt(queries.shape[-1])
            self_attention = torch.softmax(torch.matmul(queries, keys) / scale, dim=-1)
            aspect_attention = torch.tanh(torch.matmul(aspect_queries, opinion_keys) / scale)

            distances = self._tree_distances(word_pair_synpost)
            distance_masks = torch.stack(
                [(distances > cutoff).to(hidden.dtype) for cutoff in range(1, self.attention_heads + 1)],
                dim=1,
            )
            semantic_scores = self_attention + aspect_attention + distance_masks
            semantic_scores = semantic_scores.masked_fill(
                ~pair_mask[:, None, :, :], torch.finfo(semantic_scores.dtype).min
            )
            semantic_attention = torch.softmax(semantic_scores, dim=-1).mean(dim=1)
            semantic_attention = semantic_attention * pair_mask.to(hidden.dtype)

            aspect_nodes = F.relu(self.ap_fc(hidden))
            opinion_nodes = F.relu(self.op_fc(hidden))
            syntax_nodes = torch.matmul(syntax_attention, aspect_nodes)
            semantic_nodes = torch.matmul(semantic_attention, opinion_nodes)
            affine_nodes = self.ssgcn_affine(syntax_nodes)
            interaction_score = torch.matmul(affine_nodes, semantic_nodes.transpose(1, 2))
            interaction_logits = self.ssgcn_edge_classifier(interaction_score.unsqueeze(-1))

            relation_logits = [
                interaction_logits,
                relative_logits,
                dependency_logits,
                pos_pair_logits,
                tree_distance_logits,
            ]
            relation_probabilities = [torch.softmax(value, dim=-1) for value in relation_logits]
            relation_probabilities = [value * pair_mask_4d for value in relation_probabilities]
            adjacency_probabilities = torch.cat(relation_probabilities, dim=-1)
            adjacency_logits = torch.cat(relation_logits, dim=-1)

            graph_input = F.relu(self.dense(hidden))
            self_loop = torch.eye(seq_len, device=hidden.device, dtype=hidden.dtype)
            self_loop = (
                self_loop.view(1, 1, seq_len, seq_len)
                .expand(batch, 5 * int(self.config.output_dim), seq_len, seq_len)
                * pair_mask[:, None, :, :].to(hidden.dtype)
            )
            graph_nodes, refined_edges = self.gcn_layers[0](
                adjacency_probabilities, adjacency_logits, graph_input, self_loop
            )
            return [*relation_logits, refined_edges]

    SSGCN.__name__ = "SSGCN"
    SSGCN.__qualname__ = "SSGCN"
    SSGCN.__module__ = __name__
    globals()["SSGCN"] = SSGCN
    return SSGCN


def __getattr__(name: str):
    # PyABSA pickles the configured model class beside its checkpoint. Resolve
    # the torch-dependent model lazily when that pickle is loaded.
    if name == "SSGCN":
        return build_ssgcn_class()
    raise AttributeError(name)
