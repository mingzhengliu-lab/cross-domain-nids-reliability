"""Machine-readable method metadata for the LODO NIDS benchmark.

The method keys stay short and stable because they are used in paths and CLI
arguments. The display names and variant notes describe what is actually run in
this repository.
"""

from __future__ import annotations

from copy import deepcopy


ALGORITHM_METADATA: dict[str, dict] = {
    "erm": {
        "display_name": "ERM-cw",
        "paper_family": "Empirical Risk Minimization baseline",
        "implementation_variant": "class/domain-weighted ERM baseline",
        "fidelity": "project baseline",
        "claim_scope": (
            "Uses per-domain class-weighted cross entropy averaged across source "
            "domains. It is a controlled NIDS baseline, not unweighted ERM."
        ),
    },
    "irm": {
        "display_name": "IRM",
        "paper_family": "Invariant Risk Minimization",
        "implementation_variant": "DomainBed-style scalar-scale IRM penalty",
        "fidelity": "benchmark implementation",
        "claim_scope": (
            "Runs the common DG benchmark variant with source-only validation "
            "selection under the shared NIDS protocol."
        ),
    },
    "coral": {
        "display_name": "CORAL",
        "paper_family": "Deep CORAL",
        "implementation_variant": "multi-source feature mean/covariance alignment",
        "fidelity": "benchmark implementation",
        "claim_scope": (
            "Uses a standard CORAL feature-alignment penalty across source domains."
        ),
    },
    "deepcoral": {
        "display_name": "CORAL",
        "paper_family": "Deep CORAL",
        "implementation_variant": "alias of coral",
        "fidelity": "benchmark implementation",
        "claim_scope": "Kept as a CLI alias for coral.",
    },
    "groupdro": {
        "display_name": "GroupDRO",
        "paper_family": "Group Distributionally Robust Optimization",
        "implementation_variant": "source-domain groups with exponentiated weights",
        "fidelity": "benchmark implementation",
        "claim_scope": (
            "Groups are source domains in the LODO protocol, selected without "
            "target-domain data."
        ),
    },
    "dann": {
        "display_name": "DANN-source",
        "paper_family": "Domain-Adversarial Neural Networks",
        "implementation_variant": "source-only multi-source domain adversary",
        "fidelity": "adapted benchmark variant",
        "claim_scope": (
            "No unlabeled target samples are used. This should be described as a "
            "source-only DANN-style DG baseline, not the original UDA setting."
        ),
    },
    "fishr": {
        "display_name": "Fishr-lite",
        "paper_family": "Fishr",
        "implementation_variant": (
            "classifier-head per-sample gradient-variance matching with NIDS "
            "class-weighted gradients"
        ),
        "fidelity": "adapted benchmark variant",
        "claim_scope": (
            "This avoids BackPACK/full-network per-sample gradients for speed. "
            "Report as Fishr-lite unless an official Fishr implementation is run."
        ),
    },
    "boda": {
        "display_name": "BoDA-style",
        "paper_family": "BoDA / multi-domain long-tailed recognition",
        "implementation_variant": (
            "same-class cross-domain EMA prototype pull for source-only tabular NIDS"
        ),
        "fidelity": "inspired benchmark variant",
        "claim_scope": (
            "This captures balanced domain-class prototype alignment but is not the "
            "full BoDA training recipe from the image-recognition codebase."
        ),
    },
    "multitask_erm": {
        "display_name": "MultitaskERM",
        "paper_family": "Auxiliary attack-family regularized ERM",
        "implementation_variant": (
            "class/domain-weighted ERM with a shared backbone and an auxiliary "
            "source attack-family prediction head"
        ),
        "fidelity": "project diagnostic variant",
        "claim_scope": (
            "Uses only source-domain labels already present in the processed data. "
            "The final decision head remains binary attack-vs-benign; attack-family "
            "supervision is an auxiliary representation regularizer."
        ),
    },
    "multitask_erm_shared4": {
        "display_name": "Shared4-MultitaskERM",
        "paper_family": "Coarse attack-family auxiliary ERM",
        "implementation_variant": (
            "class/domain-weighted ERM with a shared four-class attack-family "
            "auxiliary head (benign, DoS/DDoS, reconnaissance/scanning, residual attack)"
        ),
        "fidelity": "project diagnostic candidate",
        "claim_scope": (
            "The family head is a source-only auxiliary regularizer; the binary "
            "attack score remains the reported prediction and the residual family "
            "is not a claim of semantic equivalence across datasets."
        ),
    },
}


def canonical_method_name(name: str) -> str:
    name = name.lower()
    if name == "deepcoral":
        return "coral"
    return name


def get_algorithm_metadata(name: str) -> dict:
    key = canonical_method_name(name)
    if key not in ALGORITHM_METADATA:
        raise KeyError(f"unknown algorithm metadata key: {name}")
    out = deepcopy(ALGORITHM_METADATA[key])
    out["method"] = key
    out["canonical_method"] = key
    return out


def method_display_name(name: str) -> str:
    return get_algorithm_metadata(name)["display_name"]
