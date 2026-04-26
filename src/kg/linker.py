"""
Lightweight n-gram entity linker:
  text → list of matched KG CUIs
"""

import re
from typing import List, Dict, Tuple
import networkx as nx


class EntityLinker:
    """
    Match n-gram spans in text to KG nodes via a lexical index.
    Longer matches win (greedy from left).
    """

    def __init__(self, lexical_index: Dict[str, str],
                 max_ngram: int = 5, max_entities: int = 10):
        self.lex          = lexical_index    # surface_lower → CUI
        self.max_ngram    = max_ngram
        self.max_entities = max_entities

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9\-]+", text.lower())

    def link(self, text: str) -> List[str]:
        """Return list of CUIs found in text (up to max_entities)."""
        tokens  = self._tokenize(text)
        linked  = []
        visited = set()
        i = 0
        while i < len(tokens) and len(linked) < self.max_entities:
            matched = False
            for n in range(min(self.max_ngram, len(tokens) - i), 0, -1):
                span = " ".join(tokens[i : i + n])
                if span in self.lex:
                    cui = self.lex[span]
                    if cui not in visited:
                        linked.append(cui)
                        visited.add(cui)
                    i += n
                    matched = True
                    break
            if not matched:
                i += 1
        return linked