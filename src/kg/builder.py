"""
Build biomedical KG from UMLS 2025AB files.

Supported inputs:
  Option A (recommended): Only MRCONSO.RRF + MRREL.RRF  ← from the 472 MB zip
  Option B:               Full UMLS release META/ folder ← from the 5.2 GB zip
  Option C:               Synthetic fallback             ← no UMLS needed (testing only)

MRCONSO.RRF column layout (19 columns, pipe-separated):
  0  CUI   – Concept Unique Identifier
  1  LAT   – Language
  2  TS    – Term Status (P = preferred)
  3  LUI   – Lexical Unique Identifier
  4  STT   – String Type
  5  SUI   – String Unique Identifier
  6  ISPREF– Is Preferred (Y/N)
  7  AUI   – Atom Unique Identifier
  8  SAUI  – Source-Asserted Atom Identifier
  9  SCUI  – Source-Asserted Concept Identifier
  10 SDUI  – Source-Asserted Descriptor Identifier
  11 SAB   – Source Abbreviation (e.g. SNOMEDCT_US, ICD10CM)
  12 TTY   – Term Type
  13 CODE  – Source-Asserted Code
  14 STR   – String (the actual name/term)  ← we want this
  15 SRL   – Source Restriction Level
  16 SUPPRESS – Suppressible flag (N = keep)
  17 CVF   – Content View Flag
  (18 = trailing empty field after last pipe)

MRREL.RRF column layout (17 columns):
  0  CUI1  – First concept CUI
  1  AUI1  – First atom AUI
  2  STYPE1– ID type for first element
  3  REL   – Relationship label (RB, RN, RO, PAR, CHD, etc.)
  4  CUI2  – Second concept CUI
  5  AUI2  – Second atom AUI
  6  STYPE2– ID type for second element
  7  RELA  – Relationship attribute (e.g. "isa", "inverse_isa")
  8  RUI   – Relationship Unique Identifier
  9  SRUI  – Source-Asserted Relationship Identifier
  10 SAB   – Source Abbreviation
  11 SL    – Source of Relationship Label
  12 RG    – Relationship Group
  13 DIR   – Directionality
  14 SUPPRESS – Suppressible flag
  15 CVF   – Content View Flag
  (16 = trailing empty field)

Usage
-----
  # With MRCONSO.RRF + MRREL.RRF files:
  export UMLS_MRCONSO=/path/to/MRCONSO.RRF
  export UMLS_MRREL=/path/to/MRREL.RRF    # optional; adds relation edges

  # Or with full META/ directory:
  export UMLS_DIR=/path/to/2025AB/META

  python scripts/02_build_kg.py
"""

import os, pickle, pathlib
from collections import Counter
import networkx as nx
from loguru import logger
from tqdm import tqdm


# ── Column indices ────────────────────────────────────────────────────────────
# MRCONSO.RRF
COL_CUI     = 0
COL_LAT     = 1
COL_TS      = 2
COL_ISPREF  = 6
COL_SAB     = 11
COL_STR     = 14
COL_SUPPRESS_CONSO = 16

# MRREL.RRF
COL_CUI1    = 0
COL_REL     = 3
COL_CUI2    = 4
COL_RELA    = 7
COL_SUPPRESS_REL = 14

# ── Filters ──────────────────────────────────────────────────────────────────
# Source vocabularies most relevant to medical QA.
SOURCES_KEEP = {
    "SNOMEDCT_US",   # SNOMED Clinical Terms
    "ICD10CM",       # ICD-10 Clinical Modification
    "ICD10",         # ICD-10
    "RXNORM",        # Drug names
    "MSH",           # MeSH (Medical Subject Headings)
    "NCI",           # NCI Thesaurus
    "HPO",           # Human Phenotype Ontology
    "GO",            # Gene Ontology
    "DRUGBANK",      # DrugBank
    "MEDLINEPLUS",   # MedlinePlus
}

# REL types to keep from MRREL.
#
# PRIOR BUG (now fixed): keeping ONLY {RB, RN, PAR, CHD} silently dropped every
# clinically meaningful UMLS edge, because clinical RELAs such as ``may_treat``,
# ``has_mechanism_of_action`` and ``contraindicated_with_disease`` all live on
# ``REL='RO'`` (Related Other) rows. Filtering them out left only SNOMED isa/
# inverse_isa taxonomy, which is why the KG branch was always negligible in
# rank-data and downstream evaluation.
#
# We now do NOT filter on REL at all (empty set = inert filter, see the loop
# below). The clinical-quality gate is enforced entirely on the RELA allow-list
# below, which is the right granularity (RELA names the semantic edge type).
RELATIONS_KEEP: set[str] = set()

# RELA allow-list aligned with the QA types declared in configs/base.yaml:
#   ["diagnosis", "treatment", "mechanism", "contraindication",
#    "factoid", "yesno", "other"]
#
# We keep three families of edges:
#   (a) taxonomy backbone — ``isa`` / ``inverse_isa`` plus equivalent broader/
#       narrower terms. These give KG distances a "shortest hop through
#       SNOMED" baseline and matter for factoid / yesno / other.
#   (b) anatomy + symptom links — finding-site, manifestation, morphology.
#       Needed for diagnosis-type questions ("which disease causes X?").
#   (c) drug + therapy + mechanism + contraindication links. Needed for
#       treatment / mechanism / contraindication questions.
#
# Generic / weak RELAs ("classifies", "mapped_to", "associated_with", etc.) are
# intentionally excluded — they flood the graph with low-signal edges that
# dilute path-distance features.
RELA_KEEP = {
    # (a) Taxonomy / equivalence
    "isa", "inverse_isa",
    "has_member", "member_of",
    "has_part", "part_of",
    "has_class", "class_of",
    "has_form", "form_of",
    "has_tradename", "tradename_of",

    # (b) Diagnosis — symptoms / findings / morphology / anatomy
    "has_finding_site", "finding_site_of",
    "has_manifestation", "manifestation_of",
    "has_associated_morphology", "associated_morphology_of",
    "has_definitional_manifestation", "definitional_manifestation_of",
    "has_clinical_course", "clinical_course_of",
    "has_pathological_process", "pathological_process_of",
    "disease_has_finding", "disease_has_associated_disease",
    "disease_has_normal_tissue_origin",
    "disease_has_abnormal_cell",
    "disease_may_have_finding",
    "may_be_diagnosed_by", "diagnoses",

    # (c) Treatment / drug-disease therapeutic links
    "treats", "treated_by",
    "may_treat", "may_be_treated_by",
    "has_therapeutic_class", "therapeutic_class_of",
    "has_dose_form", "dose_form_of",
    "has_ingredient", "ingredient_of",
    "active_ingredient_of", "has_active_ingredient",
    "precise_ingredient_of", "has_precise_ingredient",
    "has_drug_class_membership",
    "may_prevent", "may_be_prevented_by",

    # (d) Mechanism of action / targets / pharmacology
    "has_mechanism_of_action", "mechanism_of_action_of",
    "chemical_or_drug_has_mechanism_of_action",
    "has_physiologic_effect", "physiologic_effect_of",
    "has_target", "target_of",
    "has_molecular_action", "molecular_action_of",
    "has_pharmacokinetics", "pharmacokinetics_of",
    "regulates", "regulated_by",
    "inhibits", "inhibited_by",
    "induces", "induced_by",

    # (e) Contraindications + adverse interactions
    "has_contraindication", "contraindication_of",
    "contraindicated_with_disease", "contraindicated_drug",
    "has_contraindicated_drug",
    "has_contraindicated_class", "contraindicating_class_of",
    "has_adverse_effect", "adverse_effect_of",
    "interacts_with",

    # (f) Causal / pathophysiology (used by mechanism + diagnosis questions)
    "cause_of", "has_cause",
    "has_causative_agent", "causative_agent_of",
    "due_to", "underlies",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def _atom_priority(ts: str, ispref: str) -> int:
    """Lower is better. 0 = preferred-of-preferred-source atom.

    Used to deduplicate surface-form -> CUI when multiple CUIs share a string.
    """
    if ts == "P" and ispref == "Y":
        return 0   # gold: preferred atom of preferred source
    if ts == "P":
        return 1   # preferred atom (any source rank)
    if ispref == "Y":
        return 2   # preferred string-of-source
    return 3       # ordinary synonym


def build_kg_from_rrf(
    mrconso_path: str,
    out_pkl: str,
    out_lex: str,
    mrrel_path: str = None,
    max_concepts: int | None = None,
    sources_keep=None,
    relations_keep=None,
    rela_keep=None,
):
    """
    Parse MRCONSO.RRF (and optionally MRREL.RRF) to build a NetworkX DiGraph
    and a lexical index {surface_form_lower → CUI}.

    Parameters
    ----------
    mrconso_path : path to MRCONSO.RRF
    out_pkl      : output path for pickled nx.DiGraph
    out_lex      : output path for pickled lexical dict
    mrrel_path   : path to MRREL.RRF (optional; adds relation edges)
    max_concepts : cap on number of CUIs to keep (default ``None`` → unlimited)
    """
    mrconso_path = pathlib.Path(mrconso_path)
    assert mrconso_path.exists(), f"MRCONSO.RRF not found: {mrconso_path}"

    cui_label: dict[str, str] = {}            # CUI → preferred English label
    cui_alts:  dict[str, list] = {}           # CUI → all English surface forms
    lexical:   dict[str, str]  = {}           # surface_lower → CUI
    lex_prio:  dict[str, int]  = {}           # surface_lower → priority of stored CUI

    allowed_sources = SOURCES_KEEP if sources_keep is None else set(sources_keep)
    allowed_relations = RELATIONS_KEEP if relations_keep is None else set(relations_keep)
    allowed_rela = RELA_KEEP if rela_keep is None else set(rela_keep)

    logger.info(f"Parsing MRCONSO.RRF from {mrconso_path} …")
    if allowed_sources:
        logger.info(f"  Filtering to sources: {sorted(allowed_sources)}")
    else:
        logger.info("  Source filter disabled: keeping all vocabularies")
    if max_concepts:
        logger.info(f"  max_concepts cap: {max_concepts:,}")
    else:
        logger.info("  max_concepts cap: <unlimited>")

    with open(mrconso_path, encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="MRCONSO", unit=" lines"):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 17:
                continue

            # Language filter: English only
            if parts[COL_LAT] != "ENG":
                continue

            # Suppressible filter: skip suppressed atoms
            if parts[COL_SUPPRESS_CONSO] in ("O", "E", "Y"):
                continue

            # Source filter
            if allowed_sources and parts[COL_SAB] not in allowed_sources:
                continue

            cui  = parts[COL_CUI]
            term = parts[COL_STR].strip()
            ts   = parts[COL_TS]       # P = preferred, S = synonymous
            pref = parts[COL_ISPREF]   # Y = preferred atom in source

            if not term:
                continue

            # Priority-aware lexical index. When several CUIs share the same
            # surface form (e.g. "anemia") we keep the CUI from the highest-
            # quality atom (preferred-of-preferred-source first). This avoids
            # the previous "first row wins" bias which depended on MRCONSO file
            # ordering rather than concept salience.
            surface = term.lower()
            prio = _atom_priority(ts, pref)
            cur_prio = lex_prio.get(surface)
            if cur_prio is None or prio < cur_prio:
                lexical[surface] = cui
                lex_prio[surface] = prio

            # Track preferred label (TS=P and ISPREF=Y is the "best" label)
            if cui not in cui_label:
                cui_label[cui] = term
            elif ts == "P" and pref == "Y":
                cui_label[cui] = term   # upgrade to preferred form

            # Accumulate synonyms
            cui_alts.setdefault(cui, []).append(term)

            if max_concepts and len(cui_label) >= max_concepts:
                logger.info(f"  Reached max_concepts={max_concepts}, stopping MRCONSO parse.")
                break

    logger.info(f"  → {len(cui_label):,} CUIs | {len(lexical):,} surface forms")

    # ── Build graph ──────────────────────────────────────────────────────────
    logger.info("Building NetworkX DiGraph …")
    G = nx.DiGraph()
    for cui, label in cui_label.items():
        G.add_node(cui, label=label, synonyms=cui_alts.get(cui, []))

    edges_added = 0
    rela_kept_counts: Counter = Counter()
    rel_kept_counts: Counter = Counter()

    if mrrel_path:
        mrrel_path = pathlib.Path(mrrel_path)
        if not mrrel_path.exists():
            logger.warning(f"MRREL.RRF not found at {mrrel_path} — graph will have no edges.")
        else:
            logger.info(f"Parsing MRREL.RRF from {mrrel_path} …")
            if allowed_relations:
                logger.info(f"  REL filter:  {sorted(allowed_relations)}")
            else:
                logger.info("  REL filter:  <disabled> (gate is RELA-only)")
            logger.info(f"  RELA filter: {len(allowed_rela)} allowed types")
            with open(mrrel_path, encoding="utf-8", errors="replace") as f:
                for line in tqdm(f, desc="MRREL", unit=" lines"):
                    parts = line.rstrip("\n").split("|")
                    if len(parts) < 15:
                        continue

                    # Suppressible filter
                    if parts[COL_SUPPRESS_REL] in ("O", "E", "Y"):
                        continue

                    cui1 = parts[COL_CUI1]
                    cui2 = parts[COL_CUI2]
                    rel  = parts[COL_REL]
                    rela = parts[COL_RELA]

                    # Both CUIs must be in our concept set
                    if cui1 not in G or cui2 not in G:
                        continue
                    if allowed_relations and rel not in allowed_relations:
                        continue
                    if allowed_rela and rela not in allowed_rela:
                        continue

                    G.add_edge(cui1, cui2, rel=rel, rela=rela)
                    edges_added += 1
                    rela_kept_counts[rela] += 1
                    rel_kept_counts[rel] += 1

            logger.info(f"  → {edges_added:,} edges added")
            # Print a sanity-check histogram so we can immediately see whether
            # the clinical RELAs actually survived the filter.
            top_rela = rela_kept_counts.most_common(20)
            if top_rela:
                logger.info("  Top RELA (kept) | " + " ".join(f"{r}={n:,}" for r, n in top_rela))
            top_rel = rel_kept_counts.most_common(8)
            if top_rel:
                logger.info("  Top REL  (kept) | " + " ".join(f"{r}={n:,}" for r, n in top_rel))
    else:
        logger.warning("No MRREL.RRF provided — KG will have nodes but no edges. "
                       "Graph features will use entity overlap only, not path distances.")

    logger.info(f"Final graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ── Save ─────────────────────────────────────────────────────────────────
    pathlib.Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(G, f, protocol=4)
    with open(out_lex, "wb") as f:
        pickle.dump(lexical, f, protocol=4)
    logger.success(f"KG graph   → {out_pkl}")
    logger.success(f"Lexical idx→ {out_lex}")
    return G, lexical


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build from META/ directory (Full Release or Level-0 subset)
# ─────────────────────────────────────────────────────────────────────────────

def build_kg_from_umls_dir(
    umls_meta_dir: str,
    out_pkl: str,
    out_lex: str,
    max_concepts: int | None = None,
    sources_keep=None,
    relations_keep=None,
    rela_keep=None,
):
    """
    Given the META/ folder from a Full UMLS release, locate MRCONSO.RRF
    and MRREL.RRF automatically and call build_kg_from_rrf().
    """
    meta = pathlib.Path(umls_meta_dir)
    mrconso = meta / "MRCONSO.RRF"
    mrrel   = meta / "MRREL.RRF"

    if not mrconso.exists():
        raise FileNotFoundError(f"MRCONSO.RRF not found in {meta}")

    return build_kg_from_rrf(
        mrconso_path = str(mrconso),
        mrrel_path   = str(mrrel) if mrrel.exists() else None,
        out_pkl      = out_pkl,
        out_lex      = out_lex,
        max_concepts = max_concepts,
        sources_keep = sources_keep,
        relations_keep = relations_keep,
        rela_keep = rela_keep,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build directly from the 2025AB MRCONSO zip (no full install)
# ─────────────────────────────────────────────────────────────────────────────

def build_kg_from_mrconso_zip(
    zip_path: str,
    out_pkl: str,
    out_lex: str,
    max_concepts: int | None = None,
    sources_keep=None,
):
    """
    Parse MRCONSO.RRF directly from the downloaded
    umls-2025AB-mrconso.zip (472 MB) without extracting to disk.

    NOTE: Relations (MRREL) are NOT included in the MRCONSO-only zip.
          The graph will have nodes + lexical index but no edges.
          For full graph features, also extract MRREL.RRF from the
          Level-0 subset or Full Release zip.
    """
    import zipfile, io

    logger.info(f"Reading MRCONSO.RRF from zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip contains MRCONSO.RRF at root level
        names = zf.namelist()
        mrconso_name = next((n for n in names if n.endswith("MRCONSO.RRF")), None)
        if not mrconso_name:
            raise FileNotFoundError(f"MRCONSO.RRF not found inside {zip_path}. "
                                    f"Files found: {names[:10]}")

        logger.info(f"  Found {mrconso_name} inside zip")
        # Extract to a temp file so tqdm works properly
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".RRF", delete=False) as tmp:
            tmp_path = tmp.name
            logger.info(f"  Extracting to temp file {tmp_path} …")
            with zf.open(mrconso_name) as src:
                import shutil
                shutil.copyfileobj(src, tmp)

    try:
        return build_kg_from_rrf(
            mrconso_path = tmp_path,
            mrrel_path   = None,
            out_pkl      = out_pkl,
            out_lex      = out_lex,
            max_concepts = max_concepts,
            sources_keep = sources_keep,
        )
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback (no UMLS)
# ─────────────────────────────────────────────────────────────────────────────

def build_synthetic_kg(out_pkl: str, out_lex: str):
    """Minimal test KG — use only for pipeline smoke-testing."""
    logger.warning("Using SYNTHETIC KG. For real experiments provide UMLS 2025AB.")
    G = nx.DiGraph()
    triples = [
        ("C0027051", "treats",    "C0004057"),
        ("C0004057", "isa",       "C0003364"),
        ("C0027051", "has_finding_site", "C0018787"),
        ("C0037284", "isa",       "C0027051"),
        ("C0020538", "causes",    "C0018799"),
        ("C0031117", "isa",       "C0027051"),
        ("C0003864", "treats",    "C0020538"),
        ("C0009782", "treats",    "C0003864"),
    ]
    labels = {
        "C0027051": "myocardial infarction",
        "C0004057": "aspirin",
        "C0003364": "analgesics",
        "C0018787": "heart",
        "C0037284": "ST elevation MI",
        "C0020538": "hypertension",
        "C0018799": "heart failure",
        "C0031117": "NSTEMI",
        "C0003864": "arthritis",
        "C0009782": "corticosteroids",
    }
    for cui, label in labels.items():
        G.add_node(cui, label=label, synonyms=[label])
    for (s, r, t) in triples:
        G.add_edge(s, t, rel="RO", rela=r)

    lexical = {v.lower(): k for k, v in labels.items()}
    pathlib.Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(G, f, protocol=4)
    with open(out_lex, "wb") as f:
        pickle.dump(lexical, f, protocol=4)
    logger.success("Synthetic KG written.")
    return G, lexical


def load_kg(kg_pkl: str, lex_pkl: str):
    with open(kg_pkl, "rb") as f:
        G = pickle.load(f)
    with open(lex_pkl, "rb") as f:
        lex = pickle.load(f)
    logger.info(f"KG loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, "
                f"{len(lex):,} surface forms")
    return G, lex