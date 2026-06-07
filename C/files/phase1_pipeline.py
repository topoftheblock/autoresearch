"""
phase1_pipeline.py
==================
Phase 1: Structural Classification of program.md Files

Steps
-----
1. Parse every .md file in PROGRAMS_DIR into a section tree using mistletoe.
2. Assign a semantic type τ to each top-level section via heading lexicon rules.
3. Extract a syntactic feature vector φ(p) per section type.
4. Compute the type-signature σ(p) = frozenset of types present.
5. Assign each file to its ~σ class (coarsest) and its ~φ cell (medium).
6. Compute pairwise ZSS tree-edit distances for the ~T view.
7. Output: classification table (TSV) + distance matrix (TSV) + per-file JSON records.
"""

import re
import json
import math
import itertools
from pathlib import Path
from collections import defaultdict

import mistletoe
from mistletoe import Document
from mistletoe.ast_renderer import AstRenderer
import zss

# ── Configuration ─────────────────────────────────────────────────────────────

PROGRAMS_DIR = Path("/home/claude/phase1/programs")
OUTPUT_DIR   = Path("/home/claude/phase1/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tolerance ε for ~φ feature-cell binning (per feature, 0 = exact)
EPSILON = 0.0   # start at 0; increase to merge cells

# ── Section-type lexicon ──────────────────────────────────────────────────────
# Map heading text patterns → canonical type τ.
# Rules are tried in order; first match wins.

HEADING_RULES = [
    # type            regex pattern (case-insensitive, matched on heading text)
    ("setup",         r"\bsetup\b|\binitialization\b|\bprepare\b|\bpreparation\b"),
    ("scope",         r"\bscope\b|\bexperimentation\b|\bwhat you may\b|\ballowed\b|\bin.scope\b"),
    ("constraint",    r"\bconstraint\b|\bwhat you must not\b|\bforbidden\b|\bout.of.scope\b|\bdo not\b"),
    ("strategy",      r"\bstrategy\b|\bapproach\b|\bguidance\b|\boverview\b"),
    ("eval",          r"\beval\b|\bevaluation\b|\bobjective\b|\bmetric\b|\bcriterion\b"),
    ("output",        r"\boutput\b|\bformat\b|\bprint\b|\bextract\b"),
    ("logging",       r"\blogging\b|\blog\b|\bresults\b|\brecord\b"),
    ("loop",          r"\bloop\b|\bexperiment loop\b|\bcycle\b|\biterations?\b"),
    ("budget",        r"\bbudget\b|\btracking\b|\bcheckpoint\b"),
    ("meta",          r".*"),  # catch-all
]

def classify_heading(text: str) -> str:
    text = text.lower().strip()
    for type_name, pattern in HEADING_RULES:
        if re.search(pattern, text):
            return type_name
    return "meta"

# ── AST helpers ───────────────────────────────────────────────────────────────

def parse_md(path: Path) -> dict:
    """Return mistletoe AST dict for a markdown file."""
    with open(path, "r") as f:
        with AstRenderer() as renderer:
            doc = Document(f.read())
            return json.loads(renderer.render(doc))

def extract_sections(ast: dict) -> list[dict]:
    """
    Walk the AST and collect top-level sections as dicts:
      {heading_text, heading_level, type, raw_text, children_text}
    A 'section' is any Heading node together with all subsequent non-heading
    children until the next heading of equal or lower level.
    """
    children = ast.get("children", [])
    sections = []
    current = None

    for node in children:
        nt = node.get("type", "")
        if nt == "Heading":
            # Flush previous section
            if current is not None:
                sections.append(current)
            level = node.get("level", 1)
            heading_text = _node_text(node)
            current = {
                "heading_text": heading_text,
                "heading_level": level,
                "type": classify_heading(heading_text),
                "body_nodes": [],
            }
        else:
            if current is not None:
                current["body_nodes"].append(node)

    if current is not None:
        sections.append(current)

    # Add plain text of body to each section
    for sec in sections:
        sec["body_text"] = " ".join(_node_text(n) for n in sec["body_nodes"])

    return sections

def _node_text(node: dict) -> str:
    """Recursively extract all text from an AST node."""
    if node.get("type") == "RawText":
        return node.get("content", "")
    parts = []
    for child in node.get("children", []):
        parts.append(_node_text(child))
    return " ".join(parts)

# ── Feature extraction ────────────────────────────────────────────────────────

# Patterns for feature extraction
RE_BACKTICK_ID  = re.compile(r"`([a-z_][a-z0-9_]*)`")
RE_NUMERIC_RANGE= re.compile(r"\[\s*[-\d.]+\s*,\s*[-\d.]+\s*\]")
RE_NEGATION_KW  = re.compile(r"\b(not|never|must not|do not|forbidden|prohibited|cannot)\b", re.I)
RE_PROHIBITIVE  = re.compile(r"\b(do not|must not|never|forbidden|not allowed)\b", re.I)
RE_PERMISSIVE   = re.compile(r"\b(may|can|feel free|encouraged|welcome|allowed)\b", re.I)
RE_IMPERATIVE   = re.compile(r"^(change|set|use|run|inspect|review|edit|log|record|commit|keep|reset|verify|confirm|create)\b", re.I|re.M)
RE_ORDERING_KW  = re.compile(r"\b(first|then|next|after|before|one at a time|sequentially|in order)\b", re.I)
RE_METRIC_NAME  = re.compile(r"\b(val_loss|val_bpb|accuracy|auc|f1|precision|recall|perplexity|bpb|loss)\b", re.I)
RE_CMP_OP       = re.compile(r"(lower is better|higher is better|minimise|maximize|minimize|maximise|<|>|≤|≥)")
RE_WEIGHT_TERM  = re.compile(r"\b(weight|score|composite|weighted|lambda|alpha|0\.\d+\s*\*)\b", re.I)
RE_STEP_NUM     = re.compile(r"^\s*\d+\.\s", re.M)
RE_NEVER_STOP   = re.compile(r"NEVER STOP", re.I)
RE_TERMINATION  = re.compile(r"\b(stop after|until|budget|60 experiments|max experiments|if no improvement)\b", re.I)
RE_EXP_COUNT    = re.compile(r"\bexp_count\b|\bexperiment count\b|\bbudget\b", re.I)

def extract_features(sections: list[dict]) -> dict:
    """
    Build φ(p): a flat feature dict from all sections.
    Features are grouped by section type; absent sections contribute 0.
    Global features cover the whole document.
    """
    # Index sections by type (take union of all sections of that type)
    by_type = defaultdict(str)
    for sec in sections:
        by_type[sec["type"]] += " " + sec["body_text"]

    full_text = " ".join(s["body_text"] for s in sections)
    all_headings = " ".join(s["heading_text"] for s in sections)

    def count(pattern, text):
        return len(pattern.findall(text))

    def has(pattern, text):
        return 1 if pattern.search(text) else 0

    scope_text      = by_type.get("scope", "")
    constraint_text = by_type.get("constraint", "")
    strategy_text   = by_type.get("strategy", "")
    eval_text       = by_type.get("eval", "")
    loop_text       = by_type.get("loop", "")
    budget_text     = by_type.get("budget", "")

    # Merge scope + constraint text for documents that fold them into one section
    scope_all = scope_text + " " + by_type.get("meta", "")

    tokens = full_text.split()
    unique_tokens = set(t.lower() for t in tokens)
    ttr = len(unique_tokens) / max(len(tokens), 1)

    phi = {
        # ── Scope features ──────────────────────────────────────────
        "scope_named_ids":      count(RE_BACKTICK_ID,   scope_all),
        "scope_numeric_ranges": count(RE_NUMERIC_RANGE, scope_all),
        "scope_has_negation":   has(RE_NEGATION_KW,     scope_all),

        # ── Constraint features ──────────────────────────────────────
        "constraint_prohibitive": count(RE_PROHIBITIVE, full_text),
        "constraint_permissive":  count(RE_PERMISSIVE,  full_text),

        # ── Strategy features ────────────────────────────────────────
        "strategy_imperatives":   count(RE_IMPERATIVE,  strategy_text + " " + loop_text),
        "strategy_ordering_kw":   count(RE_ORDERING_KW, full_text),
        "strategy_has_hint":      1 if any(
            re.search(r"\bone at a time\b|\bsequential\b|\bexactly one\b", s["body_text"], re.I)
            for s in sections
        ) else 0,

        # ── Eval features ────────────────────────────────────────────
        "eval_metric_count":    len(set(RE_METRIC_NAME.findall(eval_text + " " + full_text))),
        "eval_cmp_ops":         count(RE_CMP_OP,    eval_text + " " + scope_text),
        "eval_has_composite":   has(RE_WEIGHT_TERM, eval_text + " " + full_text),
        "eval_multi_objective": 1 if count(RE_METRIC_NAME, eval_text + " " + scope_text) >= 2
                                     or has(RE_WEIGHT_TERM, eval_text) else 0,

        # ── Loop features ────────────────────────────────────────────
        "loop_step_count":         count(RE_STEP_NUM,     loop_text),
        "loop_has_never_stop":     has(RE_NEVER_STOP,     full_text),
        "loop_has_termination":    has(RE_TERMINATION,    loop_text + " " + budget_text),
        "loop_has_budget_counter": has(RE_EXP_COUNT,      full_text),

        # ── Global / verbosity features ──────────────────────────────
        "global_token_count":    len(tokens),
        "global_section_count":  len(sections),
        "global_ttr":            round(ttr, 4),
        "global_heading_count":  len(sections),
    }
    return phi

# ── Type-signature ────────────────────────────────────────────────────────────

def type_signature(sections: list[dict]) -> frozenset:
    """σ(p) = frozenset of section types present in the document."""
    return frozenset(s["type"] for s in sections)

# ── ZSS tree representation ───────────────────────────────────────────────────

class SectionNode:
    """Node for ZSS tree edit distance."""
    def __init__(self, label: str):
        self.label = label
        self.children: list["SectionNode"] = []

    def addkid(self, child: "SectionNode"):
        self.children.append(child)
        return self

    @staticmethod
    def get_children(node: "SectionNode"):
        return node.children

    @staticmethod
    def get_label(node: "SectionNode"):
        return node.label

def build_zss_tree(sections: list[dict]) -> SectionNode:
    """
    Build a ZSS tree from sections.
    Root → [Section(type)]* → [Feature bins]*
    """
    root = SectionNode("doc")
    for sec in sections:
        sec_node = SectionNode(sec["type"])
        # Add coarse feature bins as leaf children
        bt = sec["body_text"]
        sec_node.addkid(SectionNode(f"ids:{min(count_ids(bt), 5)}"))
        sec_node.addkid(SectionNode(f"neg:{1 if RE_NEGATION_KW.search(bt) else 0}"))
        sec_node.addkid(SectionNode(f"perm:{1 if RE_PERMISSIVE.search(bt) else 0}"))
        root.addkid(sec_node)
    return root

def count_ids(text: str) -> int:
    return len(RE_BACKTICK_ID.findall(text))

def zss_distance(tree_a: SectionNode, tree_b: SectionNode) -> float:
    return zss.simple_distance(
        tree_a, tree_b,
        SectionNode.get_children,
        SectionNode.get_label,
        lambda a, b: 0 if a == b else 1,
    )

# ── Feature binning for ~φ ────────────────────────────────────────────────────

def phi_cell(phi: dict, epsilon: float = EPSILON) -> tuple:
    """
    Discretise the feature vector into a cell key for ~φ.
    With ε=0 this is exact equality; with ε>0 nearby values merge.
    """
    if epsilon == 0.0:
        return tuple(sorted(phi.items()))
    else:
        binned = {}
        for k, v in phi.items():
            if isinstance(v, float):
                binned[k] = round(v / epsilon) * epsilon
            else:
                binned[k] = v
        return tuple(sorted(binned.items()))

# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline():
    md_files = sorted(PROGRAMS_DIR.glob("*.md"))
    if not md_files:
        print("No .md files found in", PROGRAMS_DIR)
        return

    records = []
    zss_trees = {}

    print(f"\n{'='*70}")
    print(f"  Phase 1 Pipeline — {len(md_files)} program.md files")
    print(f"{'='*70}\n")

    for path in md_files:
        ast      = parse_md(path)
        sections = extract_sections(ast)
        sigma    = type_signature(sections)
        phi      = extract_features(sections)
        cell     = phi_cell(phi)
        tree     = build_zss_tree(sections)

        zss_trees[path.stem] = tree

        record = {
            "file":      path.stem,
            "sigma":     sorted(sigma),
            "phi":       phi,
            "phi_cell":  str(cell),
            "sections":  [{"heading": s["heading_text"], "type": s["type"]} for s in sections],
        }
        records.append(record)

        print(f"  {path.stem}")
        print(f"    σ(p) = {sorted(sigma)}")
        print(f"    sections: {[(s['heading_text'], s['type']) for s in sections]}")
        print(f"    φ key features:")
        print(f"      scope_named_ids={phi['scope_named_ids']}  "
              f"scope_has_negation={phi['scope_has_negation']}  "
              f"strategy_has_hint={phi['strategy_has_hint']}")
        print(f"      eval_multi_objective={phi['eval_multi_objective']}  "
              f"constraint_prohibitive={phi['constraint_prohibitive']}  "
              f"loop_has_budget_counter={phi['loop_has_budget_counter']}")
        print(f"      global_token_count={phi['global_token_count']}  "
              f"global_ttr={phi['global_ttr']}")
        print()

    # ── ~σ equivalence classes ────────────────────────────────────────────────
    sigma_classes = defaultdict(list)
    for r in records:
        sigma_classes[frozenset(r["sigma"])].append(r["file"])

    print(f"\n{'─'*70}")
    print("  ~σ  TYPE-SIGNATURE CLASSES  (coarsest partition)")
    print(f"{'─'*70}")
    for i, (sig, members) in enumerate(sorted(sigma_classes.items(), key=lambda x: -len(x[1]))):
        print(f"  Class σ-{i+1:02d}  |  σ = {sorted(sig)}")
        print(f"           members: {members}")
    print()

    # ── ~φ equivalence classes ────────────────────────────────────────────────
    phi_classes = defaultdict(list)
    for r in records:
        phi_classes[r["phi_cell"]].append(r["file"])

    print(f"{'─'*70}")
    print(f"  ~φ  FEATURE-CELL CLASSES  (ε={EPSILON}, medium partition)")
    print(f"{'─'*70}")
    phi_singletons = sum(1 for v in phi_classes.values() if len(v) == 1)
    phi_merged     = sum(1 for v in phi_classes.values() if len(v) > 1)
    for i, (cell, members) in enumerate(phi_classes.items()):
        if len(members) > 1:
            print(f"  Merged cell φ-{i+1:02d}  |  members: {members}")
    print(f"  Singleton cells: {phi_singletons}  |  Merged cells: {phi_merged}")
    print(f"  Total ~φ classes: {len(phi_classes)}")
    print()

    # ── ZSS pairwise distance matrix ──────────────────────────────────────────
    names = [r["file"] for r in records]
    n = len(names)
    dist_matrix = [[0.0]*n for _ in range(n)]

    print(f"{'─'*70}")
    print("  ~T  ZSS TREE-EDIT DISTANCE MATRIX")
    print(f"{'─'*70}")

    for i, j in itertools.combinations(range(n), 2):
        d = zss_distance(zss_trees[names[i]], zss_trees[names[j]])
        dist_matrix[i][j] = d
        dist_matrix[j][i] = d

    # Print as table
    col_w = max(len(n) for n in names) + 2
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
    print("  " + header)
    for i, row_name in enumerate(names):
        row = f"  {row_name:<{col_w}}" + "".join(
            f"{dist_matrix[i][j]:>{col_w}.1f}" for j in range(n)
        )
        print(row)
    print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"{'─'*70}")
    print("  CLASSIFICATION SUMMARY")
    print(f"{'─'*70}")
    print(f"  {'File':<20} {'σ-class':<10} {'φ-cell':<10} "
          f"{'token_count':<14} {'named_ids':<12} {'has_hint':<10} "
          f"{'multi_obj':<12} {'has_budget':<12}")
    print(f"  {'─'*20} {'─'*9} {'─'*9} {'─'*13} {'─'*11} {'─'*9} {'─'*11} {'─'*11}")

    # Assign short labels to sigma classes
    sigma_label = {}
    for i, sig in enumerate(sorted(sigma_classes.keys(), key=lambda x: -len(sigma_classes[x]))):
        for member in sigma_classes[sig]:
            sigma_label[member] = f"σ-{i+1:02d}"

    phi_label = {}
    phi_idx = 0
    for cell, members in phi_classes.items():
        phi_idx += 1
        for m in members:
            phi_label[m] = f"φ-{phi_idx:02d}"

    for r in records:
        phi = r["phi"]
        print(f"  {r['file']:<20} {sigma_label[r['file']]:<10} {phi_label[r['file']]:<10} "
              f"{phi['global_token_count']:<14} {phi['scope_named_ids']:<12} "
              f"{phi['strategy_has_hint']:<10} {phi['eval_multi_objective']:<12} "
              f"{phi['loop_has_budget_counter']:<12}")
    print()

    # ── Write outputs ─────────────────────────────────────────────────────────
    # JSON records
    with open(OUTPUT_DIR / "records.json", "w") as f:
        json.dump(records, f, indent=2)

    # TSV classification table
    with open(OUTPUT_DIR / "classification.tsv", "w") as f:
        f.write("file\tsigma_class\tphi_class\tsigma\t"
                "token_count\tnamed_ids\thas_hint\tmulti_obj\t"
                "prohibitive\thas_budget\tttr\n")
        for r in records:
            phi = r["phi"]
            f.write(
                f"{r['file']}\t{sigma_label[r['file']]}\t{phi_label[r['file']]}\t"
                f"{','.join(sorted(r['sigma']))}\t"
                f"{phi['global_token_count']}\t{phi['scope_named_ids']}\t"
                f"{phi['strategy_has_hint']}\t{phi['eval_multi_objective']}\t"
                f"{phi['constraint_prohibitive']}\t{phi['loop_has_budget_counter']}\t"
                f"{phi['global_ttr']}\n"
            )

    # TSV distance matrix
    with open(OUTPUT_DIR / "zss_distances.tsv", "w") as f:
        f.write("\t" + "\t".join(names) + "\n")
        for i, row_name in enumerate(names):
            f.write(row_name + "\t" + "\t".join(
                f"{dist_matrix[i][j]:.1f}" for j in range(n)
            ) + "\n")

    print(f"  Outputs written to {OUTPUT_DIR}/")
    print(f"    records.json        — per-file AST + feature records")
    print(f"    classification.tsv  — σ-class, φ-class, feature summary")
    print(f"    zss_distances.tsv   — pairwise ZSS tree-edit distance matrix")
    print()

    # ── Factorisation check ───────────────────────────────────────────────────
    print(f"{'─'*70}")
    print("  FACTORISATION INDICATOR")
    print(f"{'─'*70}")
    n_sigma = len(sigma_classes)
    n_phi   = len(phi_classes)
    print(f"  |𝒯/~σ| = {n_sigma} type-signature classes")
    print(f"  |𝒯/~φ| = {n_phi}  feature-cell classes  (ε={EPSILON})")
    print(f"  |𝒯/~T| ≈ {n} (one per file at ε=0, before merging)")
    print()
    if n_sigma == 1:
        print("  All files share the same type-signature σ.")
        print("  Factorisation trivially holds at the coarsest level;")
        print("  the ~φ features carry all the discriminating information.")
    else:
        print(f"  {n_sigma} distinct σ-classes found.")
        print("  Check whether outcome distributions differ across σ-classes")
        print("  to determine whether the coarse partition already explains")
        print("  the variance, or whether finer ~φ features are needed.")
    print()
    print("  Run Phase 2 (empirical runs) to measure F([p]) per class.")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_pipeline()
