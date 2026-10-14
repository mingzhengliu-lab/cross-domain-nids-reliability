"""Pre-registered coarse attack-family ontology for NFv2 label audits.

The mapping is a source-label rule set. It does not inspect feature values or
prediction results and is applied to target attack names only after a model is
frozen. The exact-string vocabulary used by the existing benchmark is unchanged.
"""
from __future__ import annotations

import re


ONTOLOGY_VERSION = "nfv2_attack_family_ontology_v1"
SHARED4_ONTOLOGY_VERSION = "nfv2_attack_family_shared4_v1"
FAMILY_LABELS = (
    "benign",
    "dos_ddos",
    "reconnaissance_scanning",
    "credential_bruteforce",
    "injection",
    "botnet",
    "exploit",
    "backdoor",
    "fuzzing",
    "theft",
    "mitm",
    "ransomware",
    "other_attack",
)


def _contains_word(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?:^|[^a-z]){re.escape(word)}(?:$|[^a-z])", text)
               for word in words)


def coarse_attack_family(name: object) -> str:
    """Map a textual attack name to a fixed coarse family.

    Rules are ordered from the most specific operational categories to the
    residual ``other_attack`` class. Unknown or empty values are retained as
    ``other_attack`` rather than silently treated as benign.
    """
    text = "" if name is None else " ".join(str(name).strip().lower().split())
    if text in {"benign", "normal", "background", "normal traffic"}:
        return "benign"
    if text == "":
        return "other_attack"
    if _contains_word(text, ("ransomware", "ransom")):
        return "ransomware"
    if _contains_word(text, ("mitm", "man-in-the-middle", "man in the middle")):
        return "mitm"
    if _contains_word(text, ("injection", "inject", "sql", "xss")):
        return "injection"
    if _contains_word(text, ("brute", "bruteforce", "password", "credential", "ssh")):
        return "credential_bruteforce"
    if _contains_word(text, ("recon", "reconnaissance", "scan", "scanning")):
        return "reconnaissance_scanning"
    if _contains_word(text, ("ddos", "dos", "hoic", "hulk", "loic", "slowhttp", "slowloris")):
        return "dos_ddos"
    if _contains_word(text, ("botnet", "bot")):
        return "botnet"
    if _contains_word(text, ("exploit", "exploits")):
        return "exploit"
    if _contains_word(text, ("backdoor", "backdoors")):
        return "backdoor"
    if _contains_word(text, ("fuzz", "fuzzer", "fuzzers")):
        return "fuzzing"
    if _contains_word(text, ("theft", "exfiltration")):
        return "theft"
    return "other_attack"


def coarse_attack_family_names(names) -> list[str]:
    return [coarse_attack_family(name) for name in names]


SHARED4_LABELS = ("benign", "dos_ddos", "reconnaissance_scanning", "other_attack")


def shared4_attack_family(name: object) -> str:
    """Collapse the diagnostic ontology to families recurring across NFv2 sources."""
    family = coarse_attack_family(name)
    if family in SHARED4_LABELS:
        return family
    return "other_attack"


def shared4_attack_family_names(names) -> list[str]:
    return [shared4_attack_family(name) for name in names]
