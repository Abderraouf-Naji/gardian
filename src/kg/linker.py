"""Lightweight n-gram entity linker: text → list of matched KG CUIs."""

import re
from typing import Dict, List, Union

import networkx as nx


# A lex value can be either a single CUI (legacy) or a list/tuple of CUIs
# (newer multi-CUI lex when several concepts share a surface form).
LexValue = Union[str, List[str], tuple]


class EntityLinker:
    """Match n-gram spans in text to KG nodes via a lexical index.

    Longer matches win (greedy from left). When a single surface form maps to
    multiple CUIs (legitimate UMLS ambiguity), all of them are emitted, in
    insertion order, until ``max_entities`` is reached.
    """

    def __init__(
        self,
        lexical_index: Dict[str, LexValue],
        max_ngram: int = 5,
        max_entities: int = 16,
    ):
        self.lex = lexical_index            # surface_lower → CUI or [CUI...]
        self.max_ngram = int(max_ngram)
        self.max_entities = int(max_entities)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9\-]+", text.lower())

    @staticmethod
    def _cuis_for(value: LexValue) -> List[str]:
        if isinstance(value, str):
            return [value]
        return [str(c) for c in value if c]

    def link(self, text: str) -> List[str]:
        """Return list of CUIs found in text (up to max_entities)."""
        tokens = self._tokenize(text)
        linked: List[str] = []
        visited = set()
        i = 0
        while i < len(tokens) and len(linked) < self.max_entities:
            matched = False
            for n in range(min(self.max_ngram, len(tokens) - i), 0, -1):
                span = " ".join(tokens[i : i + n])
                if span in self.lex:
                    for cui in self._cuis_for(self.lex[span]):
                        if cui not in visited:
                            linked.append(cui)
                            visited.add(cui)
                            if len(linked) >= self.max_entities:
                                break
                    i += n
                    matched = True
                    break
            if not matched:
                i += 1
        return linked