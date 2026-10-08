"""Frozen Chemistry lexical normalization and family labels (shared by kit and CLI)."""
import re

FAMILY_RULES = [("reactant for a synthesis (retrosynthesis)", ["only correct reactant"]),
                ("distribution coefficient logD", ["logd"]),
                ("aqueous solubility", ["solubility"]),
                ("molar weight from SMILES", ["molar weight", "smiles"]),
                ("molar weight from IUPAC name", ["molar weight", "iupac"]),
                ("hydrogen-bond donors or acceptors", ["hydrogen bond"]),
                ("rotatable bonds", ["rotatable"]),
                ("reaction product from reactants and reagents", ["reactants", "reagents"])]


def normalize(description) -> str:
    """The bin text of one question: its `description` with double-quoted and single-quoted spans replaced by a
    placeholder, then every run of 12 or more characters from [A-Za-z0-9@+-[]()=#$\\/%.] between word boundaries
    replaced by the placeholder, whitespace collapsed, and the first 160 characters kept."""
    s = str(description)
    s = re.sub(r'"[^"]*"', '"<literal>"', s)
    s = re.sub(r"'[^']*'", "'<literal>'", s)
    s = re.sub(r"\b[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{12,}\b", "<literal>", s)
    return re.sub(r"\s+", " ", s).strip()[:160]


def family(bin_text: str) -> str:
    low = bin_text.lower()
    for name, words in FAMILY_RULES:
        if all(word in low for word in words):
            return name
    return "other"

