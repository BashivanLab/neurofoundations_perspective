"""
queries.py
----------
All PubMed search queries and subfield definitions.
Edit this file to add / remove paradigms or adjust search terms.

Three reserved keys per subfield:
  TOTAL  — raw subfield total (e.g. all "working memory"[tiab] papers).
            Used only for context; NOT used as a percentage denominator
            because most of those papers never name a specific paradigm.
  UNION  — OR of all paradigm terms within the subfield.
            This is the correct denominator: "studies that name at least
            one of our tracked paradigms."  Concentration percentages are
            computed relative to UNION.
  All other keys — individual paradigm queries.

All queries use [tiab] (title + abstract only), which is standard
practice for systematic reviews.
"""

# Helper that builds the UNION query automatically from a paradigm dict.
# Call build_union(subfield_dict) after defining each subfield dict,
# or supply it manually below.

def _inner(q):
    """Strip the trailing AND 'subfield'[tiab] anchor from a paradigm query
    so it can be OR-ed together, then the anchor is re-added once at the end."""
    return q


def build_union(subfield_key, paradigm_queries, anchor):
    """
    Constructs:  (paradigm_A OR paradigm_B OR ...) AND anchor[tiab]
    where each paradigm term is its left-hand side pattern only.

    For simplicity we just OR all the full paradigm queries together
    (PubMed handles the boolean correctly) and AND with the anchor.
    """
    parts = " OR ".join(f"({q})" for q in paradigm_queries.values())
    return f"({parts})"


SUBFIELDS = {

    # ------------------------------------------------------------------
    "Working Memory": {
        "TOTAL":  '"working memory"[tiab]',
        "UNION":  (
            '("n-back"[tiab] OR "digit span"[tiab] OR "complex span"[tiab] OR '
            '"operation span"[tiab] OR "reading span"[tiab] OR '
            '"change detection"[tiab] OR "spatial span"[tiab] OR "Corsi"[tiab] OR '
            '"delayed match to sample"[tiab] OR "delayed match-to-sample"[tiab] OR '
            '"Sternberg task"[tiab] OR "Sternberg paradigm"[tiab] OR '
            '"serial recall"[tiab] OR "running span"[tiab]) '
            'AND "working memory"[tiab]'
        ),
        "N-back":                  '"n-back"[tiab] AND "working memory"[tiab]',
        "Digit span":              '"digit span"[tiab] AND "working memory"[tiab]',
        "Complex span":            '("complex span"[tiab] OR "operation span"[tiab] OR "reading span"[tiab]) AND "working memory"[tiab]',
        "Change detection":        '"change detection"[tiab] AND "working memory"[tiab]',
        "Corsi / spatial span":    '("spatial span"[tiab] OR "Corsi"[tiab]) AND "working memory"[tiab]',
        "Delayed match-to-sample": '("delayed match to sample"[tiab] OR "delayed match-to-sample"[tiab]) AND "working memory"[tiab]',
        "Sternberg task":          '("Sternberg task"[tiab] OR "Sternberg paradigm"[tiab]) AND "working memory"[tiab]',
        "Serial recall":           '"serial recall"[tiab] AND "working memory"[tiab]',
        "Running span":            '"running span"[tiab] AND "working memory"[tiab]',
    },

    # ------------------------------------------------------------------
    "Decision Making": {
        "TOTAL":  '"decision making"[tiab]',
        "UNION":  (
            '("Iowa gambling"[tiab] OR "go/no-go"[tiab] OR "go no go"[tiab] OR '
            '"go-nogo"[tiab] OR "delay discounting"[tiab] OR "stop signal"[tiab] OR '
            '"two-alternative forced choice"[tiab] OR "2AFC"[tiab] OR '
            '"random dot motion"[tiab] OR "dot kinematogram"[tiab] OR '
            '"ultimatum game"[tiab] OR "multi-armed bandit"[tiab] OR "bandit task"[tiab] OR '
            '"reversal learning"[tiab] OR "probabilistic reversal"[tiab] OR '
            '"trust game"[tiab] OR "dictator game"[tiab] OR '
            '"balloon analogue risk"[tiab] OR "Cambridge gambling"[tiab] OR '
            '"Cambridge risk"[tiab]) '
            'AND "decision"[tiab]'
        ),
        "Iowa Gambling Task":      '"Iowa gambling task"[tiab] AND "decision"[tiab]',
        "Go/No-Go":                '("go/no-go"[tiab] OR "go no go"[tiab] OR "go-nogo"[tiab]) AND "decision"[tiab]',
        "Delay discounting":       '"delay discounting"[tiab] AND "decision"[tiab]',
        "Stop signal":             '"stop signal"[tiab] AND "decision"[tiab]',
        "2AFC / dot motion":       '("two-alternative forced choice"[tiab] OR "2AFC"[tiab] OR "random dot motion"[tiab] OR "dot kinematogram"[tiab]) AND "decision"[tiab]',
        "Ultimatum game":          '"ultimatum game"[tiab] AND "decision"[tiab]',
        "Multi-armed bandit":      '("multi-armed bandit"[tiab] OR "bandit task"[tiab]) AND "decision"[tiab]',
        "Reversal learning":       '("reversal learning"[tiab] OR "probabilistic reversal"[tiab]) AND "decision"[tiab]',
        "Trust game":              '"trust game"[tiab] AND "decision"[tiab]',
        "Dictator game":           '"dictator game"[tiab] AND "decision"[tiab]',
        "BART":                    '("balloon analogue risk"[tiab] OR "BART"[tiab]) AND "decision"[tiab]',
        "Cambridge Gambling Task": '("Cambridge gambling"[tiab] OR "Cambridge risk"[tiab]) AND "decision"[tiab]',
    },

    # ------------------------------------------------------------------
    "Spatial Navigation": {
        "TOTAL":  '"spatial navigation"[tiab]',
        "UNION":  (
            '("Morris water maze"[tiab] OR "radial arm maze"[tiab] OR '
            '"T-maze"[tiab] OR "Barnes maze"[tiab] OR "Y-maze"[tiab] OR '
            '"virtual navigation"[tiab] OR "virtual maze"[tiab] OR '
            '"path integration"[tiab] OR "Sea Hero Quest"[tiab]) '
            'AND ("navigation"[tiab] OR "spatial"[tiab])'
        ),
        "Morris water maze":   '"Morris water maze"[tiab] AND ("navigation"[tiab] OR "spatial"[tiab])',
        "Radial arm maze":     '"radial arm maze"[tiab] AND ("navigation"[tiab] OR "spatial"[tiab])',
        "T-maze":              '"T-maze"[tiab] AND ("navigation"[tiab] OR "spatial"[tiab])',
        "Barnes maze":         '"Barnes maze"[tiab] AND ("navigation"[tiab] OR "spatial"[tiab])',
        "Y-maze":              '"Y-maze"[tiab] AND ("navigation"[tiab] OR "spatial"[tiab])',
        "Virtual navigation":  '("virtual navigation"[tiab] OR "virtual maze"[tiab]) AND "spatial"[tiab]',
        "Open field":          '"open field"[tiab] AND "spatial"[tiab] AND ("navigation"[tiab] OR "memory"[tiab])',
        "Path integration":    '"path integration"[tiab] AND ("navigation"[tiab] OR "spatial"[tiab])',
        "Sea Hero Quest":      '"Sea Hero Quest"[tiab] AND ("navigation"[tiab] OR "spatial"[tiab])',
    },

    # ------------------------------------------------------------------
    "Attention": {
        "TOTAL":  '"attention"[tiab] AND ("task"[tiab] OR "paradigm"[tiab])',
        "UNION":  (
            '("Stroop"[tiab] OR "visual search"[tiab] OR "flanker"[tiab] OR '
            '"Eriksen"[tiab] OR "oddball"[tiab] OR '
            '"continuous performance task"[tiab] OR '
            '"attentional blink"[tiab] OR "rapid serial visual"[tiab] OR '
            '"binocular rivalry"[tiab] OR "inhibition of return"[tiab] OR '
            '"multiple object tracking"[tiab]) '
            'AND "attention"[tiab]'
        ),
        "Stroop":                    '"Stroop"[tiab] AND "attention"[tiab]',
        "Visual search":             '"visual search"[tiab] AND "attention"[tiab]',
        "Flanker (Eriksen)":         '("flanker"[tiab] OR "Eriksen"[tiab]) AND "attention"[tiab]',
        "Oddball (P300)":            '"oddball"[tiab] AND "attention"[tiab]',
        "CPT":                       '("continuous performance task"[tiab] OR "CPT"[tiab]) AND "attention"[tiab]',
        "Posner cueing":             '("Posner"[tiab] OR "spatial cueing"[tiab] OR "endogenous cue"[tiab]) AND "attention"[tiab]',
        "Attentional blink":         '("attentional blink"[tiab] OR "rapid serial visual"[tiab]) AND "attention"[tiab]',
        "Binocular rivalry":         '"binocular rivalry"[tiab] AND "attention"[tiab]',
        "Inhibition of return":      '"inhibition of return"[tiab] AND "attention"[tiab]',
        "Multiple object tracking":  '"multiple object tracking"[tiab] AND "attention"[tiab]',
    },

}

# Colour scheme for plots (one per subfield)
SUBFIELD_COLORS = {
    "Working Memory":    "#2196F3",
    "Decision Making":   "#FF5722",
    "Spatial Navigation":"#4CAF50",
    "Attention":         "#9C27B0",
}
