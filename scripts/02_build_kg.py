"""Build one or many UMLS KGs from your local UMLS data."""

import argparse
import os
import pathlib
import shutil
import sys
from typing import Iterable, List, Tuple
sys.path.insert(0, ".")
from omegaconf import OmegaConf
from loguru import logger
from src.kg.builder import (
    RELATIONS_KEEP,
    RELA_KEEP,
    SOURCES_KEEP,
    build_kg_from_rrf,
    build_kg_from_umls_dir,
    build_kg_from_mrconso_zip,
    build_synthetic_kg,
)


KG_ROOT = pathlib.Path("data/kg")
KG_DEFAULT_DIR = KG_ROOT / "default"
KG_VARIANTS_DIR = KG_ROOT / "variants"
KG_SOURCES_DIR = KG_ROOT / "sources"


def parse_args():
    p = argparse.ArgumentParser(description="Build UMLS KG artifacts.")
    p.add_argument(
        "--build-all",
        action="store_true",
        help="Build all available KG variants from the selected UMLS source.",
    )
    p.add_argument(
        "--max-concepts",
        type=int,
        default=0,
        help=(
            "Max CUIs to keep per KG variant (default 0 = unlimited). "
            "Set a positive value only for quick smoke-test builds."
        ),
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Prefix for build-all outputs, e.g. data/umls_kg",
    )
    p.add_argument(
        "--include-per-source",
        action="store_true",
        help="Also build one KG per detected UMLS source (can create many files).",
    )
    p.add_argument(
        "--skip-organize-existing",
        action="store_true",
        help="Skip reorganizing existing flat KG files in data/ before building.",
    )
    p.add_argument(
        "--organize-only",
        action="store_true",
        help="Only reorganize existing KG files in data/ and exit.",
    )
    p.add_argument(
        "--skim-extra-kg",
        action="store_true",
        help="List (graph, lexical) pairs under data/kg/sources/ and data/kg/variants/, then exit.",
    )
    p.add_argument(
        "--bootstrap-default-from-extra",
        action="store_true",
        help="Copy the largest available extra KG into data/kg/default/ (cfg paths), then exit.",
    )
    p.add_argument(
        "--no-bootstrap-from-extra",
        action="store_true",
        help="When building default without UMLS, do not copy from sources/variants/ (use synthetic only).",
    )
    p.add_argument(
        "--force-rebuild-default",
        action="store_true",
        help="Delete existing data/kg/default KG files and rebuild from current UMLS settings.",
    )
    p.add_argument(
        "--require-umls-source",
        action="store_true",
        help=(
            "Fail if no UMLS source is configured (prevents bootstrap/synthetic fallback). "
            "Use for strict clinical builds."
        ),
    )
    return p.parse_args()


def _ensure_parent(path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def _default_stem(cfg) -> str:
    return pathlib.Path(cfg.paths.kg_graph).with_suffix("").name


def _default_single_outputs(cfg) -> tuple[str, str]:
    stem = _default_stem(cfg)
    out_pkl = (KG_DEFAULT_DIR / f"{stem}.pkl").as_posix()
    out_lex = (KG_DEFAULT_DIR / f"{stem}_lex.pkl").as_posix()
    return out_pkl, out_lex


def _default_variants_prefix(cfg) -> str:
    stem = _default_stem(cfg)
    return (KG_VARIANTS_DIR / stem).as_posix()


def _organize_existing_kg_files() -> None:
    data_dir = pathlib.Path("data")
    if not data_dir.is_dir():
        return

    KG_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    KG_VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    KG_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    for p in sorted(data_dir.glob("*.pkl")):
        name = p.name
        target: pathlib.Path | None = None

        if name.startswith("umls_kg_source_"):
            source_slug = name[len("umls_kg_source_") :].removesuffix("_lex.pkl").removesuffix(".pkl")
            target = KG_SOURCES_DIR / source_slug / name
        elif name.startswith("umls_kg_") and name != "umls_kg.pkl":
            target = KG_VARIANTS_DIR / name
        elif name in {"umls_kg.pkl", "kg_lexical_index.pkl"}:
            target_name = "umls_kg_lex.pkl" if name == "kg_lexical_index.pkl" else name
            target = KG_DEFAULT_DIR / target_name

        if target is None:
            continue
        if p.resolve() == target.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            logger.warning(f"[organize] target exists, skipping move: {target}")
            skipped += 1
            continue
        shutil.move(p.as_posix(), target.as_posix())
        moved += 1
        logger.info(f"[organize] moved {p} -> {target}")

    logger.info(f"[organize] completed: moved={moved}, skipped={skipped}")


def _lex_path_for_graph(graph: pathlib.Path) -> pathlib.Path:
    """``foo.pkl`` -> ``foo_lex.pkl`` (matches build-all / per-source naming)."""
    return graph.with_name(f"{graph.stem}_lex.pkl")


def _iter_extra_kg_pairs() -> List[Tuple[pathlib.Path, pathlib.Path, str]]:
    """
    Collect complete (graph_pkl, lexical_pkl) pairs from sources/ and variants/.

    Returns list of (graph, lex, short_label) for logging.
    """
    pairs: List[Tuple[pathlib.Path, pathlib.Path, str]] = []
    if KG_SOURCES_DIR.is_dir():
        for sub in sorted(KG_SOURCES_DIR.iterdir()):
            if not sub.is_dir():
                continue
            g = sub / "umls_kg.pkl"
            l = sub / "umls_kg_lex.pkl"
            if g.is_file() and l.is_file():
                pairs.append((g, l, f"sources/{sub.name}"))
    if KG_VARIANTS_DIR.is_dir():
        for g in sorted(KG_VARIANTS_DIR.glob("*.pkl")):
            if g.name.endswith("_lex.pkl"):
                continue
            l = _lex_path_for_graph(g)
            if l.is_file():
                pairs.append((g, l, f"variants/{g.name}"))
    return pairs


def _skim_extra_kg(cfg) -> None:
    """Log all extra KG pairs (sizes) for inspection."""
    out_pkl, out_lex = _default_single_outputs(cfg)
    logger.info(f"[skim] configured default targets: {out_pkl} | {out_lex}")
    pairs = _iter_extra_kg_pairs()
    if not pairs:
        logger.warning("[skim] no complete (graph, *_lex.pkl) pairs under sources/ or variants/")
        return
    logger.info(f"[skim] found {len(pairs)} pair(s):")
    for g, l, label in sorted(pairs, key=lambda t: t[0].stat().st_size, reverse=True):
        logger.info(
            f"  - {label} | graph={g} ({g.stat().st_size:,} B) | lex={l} ({l.stat().st_size:,} B)"
        )


def _bootstrap_default_from_extra(cfg) -> bool:
    """
    If default KG files are missing, copy the largest graph (by bytes) + its lexical index.

    Returns True if ``data/kg/default/`` now has both files (either already there or just copied).
    """
    out_pkl_s, out_lex_s = _default_single_outputs(cfg)
    out_graph = pathlib.Path(out_pkl_s)
    out_lex = pathlib.Path(out_lex_s)
    if out_graph.is_file() and out_lex.is_file():
        return True
    pairs = _iter_extra_kg_pairs()
    if not pairs:
        return False
    pairs.sort(key=lambda t: t[0].stat().st_size, reverse=True)
    for g, l, label in pairs:
        logger.info(
            f"[bootstrap-default] candidate {label} | graph={g.stat().st_size:,} B | lex={l.stat().st_size:,} B"
        )
    g_src, l_src, label = pairs[0]
    KG_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(g_src, out_graph)
    shutil.copy2(l_src, out_lex)
    logger.success(
        f"[bootstrap-default] wrote {out_graph} and {out_lex} from {label} "
        f"(largest graph among {len(pairs)} pair(s))"
    )
    return True


def _build_single_default(
    cfg,
    max_concepts: int,
    *,
    allow_bootstrap_from_extra: bool = True,
    require_umls_source: bool = False,
):
    out_pkl, out_lex = _default_single_outputs(cfg)
    _ensure_parent(out_pkl)
    _ensure_parent(out_lex)
    max_concepts = None if max_concepts <= 0 else max_concepts

    mrconso_path = os.environ.get("UMLS_MRCONSO", "")
    mrrel_path = os.environ.get("UMLS_MRREL", "")
    zip_path = os.environ.get("UMLS_MRCONSO_ZIP", "")
    umls_dir = os.environ.get("UMLS_DIR", "")

    if mrconso_path and os.path.isfile(mrconso_path):
        logger.info(f"[Option 1] Using extracted MRCONSO.RRF: {mrconso_path}")
        build_kg_from_rrf(
            mrconso_path=mrconso_path,
            mrrel_path=mrrel_path if os.path.isfile(mrrel_path) else None,
            out_pkl=out_pkl,
            out_lex=out_lex,
            max_concepts=max_concepts,
            sources_keep=SOURCES_KEEP,
            relations_keep=RELATIONS_KEEP,
            rela_keep=RELA_KEEP,
        )
        return

    if zip_path and os.path.isfile(zip_path):
        logger.info(f"[Option 2] Using MRCONSO zip: {zip_path}")
        logger.warning("MRREL not available -> path-distance features will use max cap.")
        build_kg_from_mrconso_zip(
            zip_path,
            out_pkl,
            out_lex,
            max_concepts=max_concepts,
            sources_keep=SOURCES_KEEP,
        )
        return

    if umls_dir and os.path.isdir(umls_dir):
        logger.info(f"[Option 3] Using META/ directory: {umls_dir}")
        build_kg_from_umls_dir(
            umls_dir,
            out_pkl,
            out_lex,
            max_concepts=max_concepts,
            sources_keep=SOURCES_KEEP,
            relations_keep=RELATIONS_KEEP,
            rela_keep=RELA_KEEP,
        )
        return

    if pathlib.Path(out_pkl).is_file() and pathlib.Path(out_lex).is_file():
        logger.info(f"Default KG already present -> {out_pkl} ; skip synthetic/bootstrap.")
        return

    if require_umls_source:
        raise FileNotFoundError(
            "require-umls-source enabled but no UMLS source found.\n"
            "Set one of:\n"
            "  UMLS_MRCONSO=/path/to/MRCONSO.RRF (+ optional UMLS_MRREL=/path/to/MRREL.RRF)\n"
            "  UMLS_DIR=/path/to/2025AB/META\n"
            "  UMLS_MRCONSO_ZIP=/path/to/umls-2025AB-mrconso.zip (nodes-only, no MRREL edges)."
        )

    if allow_bootstrap_from_extra and _bootstrap_default_from_extra(cfg):
        return

    logger.warning(
        "No UMLS source found. Set one of:\n"
        "  UMLS_MRCONSO=/path/to/MRCONSO.RRF\n"
        "  UMLS_MRREL=/path/to/MRREL.RRF\n"
        "  UMLS_MRCONSO_ZIP=/path/to/umls-2025AB-mrconso.zip\n"
        "  UMLS_DIR=/path/to/2025AB/META\n"
        "Or populate data/kg/default/ from existing builds:\n"
        "  python3 scripts/02_build_kg.py --bootstrap-default-from-extra\n"
        "Using SYNTHETIC KG instead."
    )
    build_synthetic_kg(out_pkl, out_lex)


def _build_all_variants(cfg, max_concepts: int, out_prefix: str | None):
    max_concepts = None if max_concepts <= 0 else max_concepts
    out_prefix = out_prefix or _default_variants_prefix(cfg)
    _ensure_parent(f"{out_prefix}_x.pkl")

    umls_dir = os.environ.get("UMLS_DIR", "umls_data/META")
    meta_dir = pathlib.Path(umls_dir)
    if not meta_dir.is_dir():
        raise FileNotFoundError(
            f"build-all requires a META directory. Missing: {meta_dir}. "
            "Set UMLS_DIR=/path/to/2025AB/META."
        )

    variants = [
        {
            "name": "clinical_filtered",
            "sources_keep": SOURCES_KEEP,
            "relations_keep": RELATIONS_KEEP,
            "rela_keep": RELA_KEEP,
            "with_edges": True,
        },
        {
            "name": "all_sources_med_rel",
            "sources_keep": set(),
            "relations_keep": RELATIONS_KEEP,
            "rela_keep": RELA_KEEP,
            "with_edges": True,
        },
        {
            "name": "all_sources_all_rel",
            "sources_keep": set(),
            "relations_keep": set(),
            "rela_keep": set(),
            "with_edges": True,
        },
        {
            "name": "clinical_no_edges",
            "sources_keep": SOURCES_KEEP,
            "relations_keep": RELATIONS_KEEP,
            "rela_keep": RELA_KEEP,
            "with_edges": False,
        },
        {
            "name": "diagnosis_focus",
            "sources_keep": {"SNOMEDCT_US", "ICD10CM", "ICD10", "MSH"},
            "relations_keep": RELATIONS_KEEP,
            "rela_keep": RELA_KEEP,
            "with_edges": True,
        },
        {
            "name": "drug_focus",
            "sources_keep": {"RXNORM", "DRUGBANK", "NCI"},
            "relations_keep": RELATIONS_KEEP,
            "rela_keep": RELA_KEEP,
            "with_edges": True,
        },
        {
            "name": "phenotype_focus",
            "sources_keep": {"HPO", "GO", "NCI", "MSH"},
            "relations_keep": RELATIONS_KEEP,
            "rela_keep": RELA_KEEP,
            "with_edges": True,
        },
    ]

    logger.info(f"Building {len(variants)} KG variants from {meta_dir}")
    for spec in variants:
        out_pkl = f"{out_prefix}_{spec['name']}.pkl"
        out_lex = f"{out_prefix}_{spec['name']}_lex.pkl"
        logger.info(f"[build-all] variant={spec['name']} -> {out_pkl}")
        if spec["with_edges"]:
            build_kg_from_umls_dir(
                umls_meta_dir=str(meta_dir),
                out_pkl=out_pkl,
                out_lex=out_lex,
                max_concepts=max_concepts,
                sources_keep=spec["sources_keep"],
                relations_keep=spec["relations_keep"],
                rela_keep=spec["rela_keep"],
            )
        else:
            build_kg_from_rrf(
                mrconso_path=str(meta_dir / "MRCONSO.RRF"),
                mrrel_path=None,
                out_pkl=out_pkl,
                out_lex=out_lex,
                max_concepts=max_concepts,
                sources_keep=spec["sources_keep"],
            )


def _detect_sources(meta_dir: pathlib.Path) -> set[str]:
    mrconso = meta_dir / "MRCONSO.RRF"
    if not mrconso.exists():
        return set()
    detected: set[str] = set()
    with open(mrconso, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) > 11 and parts[11]:
                detected.add(parts[11])
    return detected


def _iter_source_specs(detected_sources: Iterable[str]):
    for sab in sorted(detected_sources):
        yield {
            "name": f"source_{sab.lower()}",
            "sources_keep": {sab},
            "relations_keep": RELATIONS_KEEP,
            "rela_keep": RELA_KEEP,
            "with_edges": True,
        }


def main():
    args = parse_args()
    cfg = OmegaConf.load("configs/base.yaml")
    if not args.skip_organize_existing:
        _organize_existing_kg_files()
    if args.organize_only:
        logger.info("Organization only mode complete.")
        return

    if args.skim_extra_kg:
        _skim_extra_kg(cfg)
        return

    if args.bootstrap_default_from_extra:
        out_pkl, out_lex = _default_single_outputs(cfg)
        if pathlib.Path(out_pkl).is_file() and pathlib.Path(out_lex).is_file():
            logger.warning(f"[bootstrap-default] already exists: {out_pkl} | {out_lex} — nothing to do")
            return
        if not _bootstrap_default_from_extra(cfg):
            raise FileNotFoundError(
                "No complete KG pairs under data/kg/sources/*/umls_kg.pkl + umls_kg_lex.pkl "
                "or data/kg/variants/*.pkl + matching *_lex.pkl."
            )
        return

    if args.build_all:
        _build_all_variants(cfg, max_concepts=args.max_concepts, out_prefix=args.out_prefix)
        if args.include_per_source:
            umls_dir = os.environ.get("UMLS_DIR", "umls_data/META")
            meta_dir = pathlib.Path(umls_dir)
            if not meta_dir.is_dir():
                raise FileNotFoundError(
                    f"--include-per-source requires META directory. Missing: {meta_dir}"
                )
            detected = _detect_sources(meta_dir)
            logger.info(f"Detected {len(detected)} UMLS sources in MRCONSO")
            for spec in _iter_source_specs(detected):
                source_name = next(iter(spec["sources_keep"])).lower()
                source_dir = KG_SOURCES_DIR / source_name
                out_pkl = (source_dir / "umls_kg.pkl").as_posix()
                out_lex = (source_dir / "umls_kg_lex.pkl").as_posix()
                _ensure_parent(out_pkl)
                _ensure_parent(out_lex)
                logger.info(f"[build-all][per-source] {spec['name']} -> {out_pkl}")
                build_kg_from_umls_dir(
                    umls_meta_dir=str(meta_dir),
                    out_pkl=out_pkl,
                    out_lex=out_lex,
                    max_concepts=None if args.max_concepts <= 0 else args.max_concepts,
                    sources_keep=spec["sources_keep"],
                    relations_keep=spec["relations_keep"],
                    rela_keep=spec["rela_keep"],
                )
    else:
        if args.force_rebuild_default:
            out_pkl, out_lex = _default_single_outputs(cfg)
            for p in (pathlib.Path(out_pkl), pathlib.Path(out_lex)):
                if p.exists():
                    p.unlink()
                    logger.info(f"[force-rebuild-default] removed {p}")
        _build_single_default(
            cfg,
            max_concepts=args.max_concepts,
            allow_bootstrap_from_extra=not args.no_bootstrap_from_extra,
            require_umls_source=bool(args.require_umls_source),
        )


if __name__ == "__main__":
    main()