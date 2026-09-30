import itertools
import json
import math
from collections import Counter
from fractions import Fraction

MAX_TRAITS = 4  # the Punnett square has 4**n cells
Y = "Y"  # the Y chromosome in an X-linked genotype, e.g. ["Xb", "Y"]

# Higher rank wins. Equal ranks between different alleles means codominance.
Y_RANK = 0
DOMINANCE_RANK = {"recessive": 1, "dominant": 2, "codominant": 2}

_ALLELE_PAIR = {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}

PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "traits": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_TRAITS,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "x_linked": {"type": "boolean"},
                    "alleles": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "phenotype": {"type": "string"},
                                "dominance": {"type": "string", "enum": list(DOMINANCE_RANK)},
                            },
                            "required": ["symbol", "phenotype", "dominance"],
                        },
                    },
                    "heterozygous_phenotype": {"type": "string"},
                    "parent1": _ALLELE_PAIR,
                    "parent2": _ALLELE_PAIR,
                },
                "required": [
                    "name", "x_linked", "alleles", "heterozygous_phenotype", "parent1", "parent2",
                ],
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["traits", "assumptions"],
}

SYSTEM_PROMPT = """\
You extract the details of a genetic cross from the user's message and answer with JSON only. \
Never predict the offspring - only describe the cross.

- "traits": one entry per trait or gene being crossed.
- "alleles": every allele of the trait with a short "symbol", the "phenotype" it produces, and its \
"dominance": "dominant", "recessive" or "codominant". Use an uppercase letter for the dominant \
allele and the same lowercase letter for the recessive one unless the message gives symbols.
- "parent1" / "parent2": exactly two allele symbols each, copied from "alleles". \
"homozygous dominant" is [A, A]; "heterozygous", "hybrid" or "carrier" is [A, a]; \
"homozygous recessive" or "pure-breeding recessive" is [a, a]. A parent showing a recessive \
phenotype is homozygous recessive.
- "heterozygous_phenotype": only for incomplete dominance, the blended phenotype of the \
heterozygote (e.g. "pink"). Mark both alleles "codominant". Otherwise "".
- "x_linked": true only for traits on the X chromosome. Write allele symbols like "XB" and "Xb" \
(never list Y as an allele). A male is [X allele, "Y"]; a female is two X alleles.
- "assumptions": anything you had to guess, e.g. a parent's genotype the message does not state. \
Empty list if nothing."""


def _allele(symbol, phenotype, dominance):
    return {"symbol": symbol, "phenotype": phenotype, "dominance": dominance}


# Few-shot examples for the prompt. Tests check each one builds a valid cross.
EXAMPLES = [
    (
        "Cross two pea plants that are both heterozygous for seed shape (round is dominant over "
        "wrinkled) and seed color (yellow is dominant over green).",
        {
            "traits": [
                {
                    "name": "seed shape",
                    "x_linked": False,
                    "alleles": [
                        _allele("R", "round", "dominant"),
                        _allele("r", "wrinkled", "recessive"),
                    ],
                    "heterozygous_phenotype": "",
                    "parent1": ["R", "r"],
                    "parent2": ["R", "r"],
                },
                {
                    "name": "seed color",
                    "x_linked": False,
                    "alleles": [
                        _allele("Y", "yellow", "dominant"),
                        _allele("y", "green", "recessive"),
                    ],
                    "heterozygous_phenotype": "",
                    "parent1": ["Y", "y"],
                    "parent2": ["Y", "y"],
                },
            ],
            "assumptions": [],
        },
    ),
    (
        "Two pink snapdragons are crossed. Red and white flowers show incomplete dominance.",
        {
            "traits": [
                {
                    "name": "flower color",
                    "x_linked": False,
                    "alleles": [
                        _allele("R", "red", "codominant"),
                        _allele("W", "white", "codominant"),
                    ],
                    "heterozygous_phenotype": "pink",
                    "parent1": ["R", "W"],
                    "parent2": ["R", "W"],
                }
            ],
            "assumptions": [],
        },
    ),
    (
        "A woman with heterozygous type A blood and a man with heterozygous type B blood have a child.",
        {
            "traits": [
                {
                    "name": "blood type",
                    "x_linked": False,
                    "alleles": [
                        _allele("IA", "A", "codominant"),
                        _allele("IB", "B", "codominant"),
                        _allele("i", "O", "recessive"),
                    ],
                    "heterozygous_phenotype": "",
                    "parent1": ["IA", "i"],
                    "parent2": ["IB", "i"],
                }
            ],
            "assumptions": [],
        },
    ),
    (
        "A colorblind man has children with a woman who is a carrier of red-green colorblindness.",
        {
            "traits": [
                {
                    "name": "color vision",
                    "x_linked": True,
                    "alleles": [
                        _allele("XB", "normal vision", "dominant"),
                        _allele("Xb", "colorblind", "recessive"),
                    ],
                    "heterozygous_phenotype": "",
                    "parent1": ["Xb", "Y"],
                    "parent2": ["XB", "Xb"],
                }
            ],
            "assumptions": [],
        },
    ),
    (
        "Cross a tall pea plant with a short pea plant. Tall is dominant.",
        {
            "traits": [
                {
                    "name": "height",
                    "x_linked": False,
                    "alleles": [
                        _allele("T", "tall", "dominant"),
                        _allele("t", "short", "recessive"),
                    ],
                    "heterozygous_phenotype": "",
                    "parent1": ["T", "T"],
                    "parent2": ["t", "t"],
                }
            ],
            "assumptions": ["The tall parent's genotype was not given, so it was assumed to be TT."],
        },
    ),
]


# --------------------------------------------------------------------------- llama.cpp


def load_model(model_path, n_ctx=4096, **kwargs):
    """Load a GGUF instruct model with llama-cpp-python. Extra kwargs go to `Llama`."""
    from llama_cpp import Llama  # imported here so the genetics functions work without it

    return Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, **kwargs)


def parse_input(llm, text):
    """Ask the model to turn a description of a cross into the PARSE_SCHEMA dict."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for question, answer in EXAMPLES:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": json.dumps(answer, separators=(",", ":"))})
    messages.append({"role": "user", "content": text})

    response = llm.create_chat_completion(
        messages=messages,
        response_format={"type": "json_object", "schema": PARSE_SCHEMA},
        temperature=0.0,
        max_tokens=1024,
    )
    return json.loads(response["choices"][0]["message"]["content"])


def predict(llm, text, attempts=3):
    """Parse `text` with the model and predict the cross.

    The grammar guarantees the JSON shape, not that it makes genetic sense (e.g. a parent
    allele missing from the allele list), so a rejected parse is retried with the error
    added to the request.
    """
    request = text
    last_error = None
    for _ in range(attempts):
        try:
            return predict_cross(build_cross(parse_input(llm, request)))
        except ValueError as error:  # includes json.JSONDecodeError from a truncated reply
            last_error = error
            request = f"{text}\n\n(Your previous answer was rejected: {error}. Fix it.)"
    raise ValueError(f"could not get a valid cross from {text!r}: {last_error}")


# --------------------------------------------------------------------------- genetics


def build_cross(parsed):
    """Validate a parsed cross (PARSE_SCHEMA shape) and normalise it for `predict_cross`."""
    raw_traits = parsed["traits"]
    if not 1 <= len(raw_traits) <= MAX_TRAITS:
        raise ValueError(f"a cross needs between 1 and {MAX_TRAITS} traits, got {len(raw_traits)}")
    traits = [_build_trait(raw) for raw in raw_traits]
    if sum(trait["x_linked"] for trait in traits) > 1:
        # X-linked genes are linked, and the sex of each offspring must agree across traits.
        raise ValueError("only one X-linked trait per cross is supported")
    return {"traits": traits, "assumptions": list(parsed.get("assumptions", []))}


def _build_trait(raw):
    name = raw["name"].strip() or "trait"
    x_linked = bool(raw["x_linked"])

    phenotypes, ranks = {}, {}
    for allele in raw["alleles"]:
        symbol = allele["symbol"].strip()
        if not symbol:
            raise ValueError(f"{name}: an allele has an empty symbol")
        if symbol in phenotypes:
            raise ValueError(f"{name}: allele {symbol!r} is listed twice")
        if x_linked and symbol == Y:
            raise ValueError(f"{name}: do not list Y as an allele of an X-linked trait")
        phenotypes[symbol] = allele["phenotype"].strip() or symbol
        ranks[symbol] = DOMINANCE_RANK[allele["dominance"]]
    if x_linked:
        ranks[Y] = Y_RANK

    trait = {
        "name": name,
        "x_linked": x_linked,
        "phenotypes": phenotypes,
        "ranks": ranks,
        # Incomplete dominance has one blended heterozygote, only well defined for two alleles.
        "blend": raw.get("heterozygous_phenotype", "").strip() if len(phenotypes) == 2 else "",
    }
    trait["parents"] = [
        _normalise(trait, _check_genotype(trait, raw[key], key)) for key in ("parent1", "parent2")
    ]
    if x_linked and all(Y in parent for parent in trait["parents"]):
        raise ValueError(f"{name}: both parents carry Y, so they cannot both be male")
    return trait


def _check_genotype(trait, genotype, key):
    genotype = [allele.strip() for allele in genotype]
    unknown = [allele for allele in genotype if allele not in trait["ranks"]]
    if len(genotype) != 2 or unknown:
        known = ", ".join(trait["ranks"])
        raise ValueError(
            f"{trait['name']}: {key} genotype {genotype} must be two of the alleles [{known}]"
        )
    if trait["x_linked"] and genotype.count(Y) > 1:
        raise ValueError(f"{trait['name']}: {key} genotype {genotype} has two Y chromosomes")
    return genotype


def _normalise(trait, genotype):
    """Order alleles most dominant first, so 'rR' and 'Rr' are the same genotype."""
    return tuple(sorted(genotype, key=lambda allele: (-trait["ranks"][allele], allele)))


def _label(alleles):
    return "".join(alleles) if all(len(a) == 1 for a in alleles) else "/".join(alleles)


def expressed_phenotype(trait, genotype):
    """The phenotype a genotype (tuple of two alleles) shows for this trait."""
    suffix = ""
    if trait["x_linked"]:
        suffix = " (male)" if Y in genotype else " (female)"
        genotype = tuple(allele for allele in genotype if allele != Y)  # a male shows his one X

    if len(set(genotype)) == 1:
        label = trait["phenotypes"][genotype[0]]
    elif trait["blend"]:
        label = trait["blend"]
    else:
        top = max(trait["ranks"][allele] for allele in genotype)
        winners = [allele for allele in genotype if trait["ranks"][allele] == top]
        # Two winners are codominant and both show. Genotypes are normalised, so order is stable.
        label = " and ".join(dict.fromkeys(trait["phenotypes"][allele] for allele in winners))
    return label + suffix


def gametes(traits, parent):
    """Every gamete of parent 0 or 1 as a tuple with one allele per trait.

    A parent contributes 2**n gametes across n traits. A homozygous parent lists the same
    gamete more than once, exactly as the rows and columns of a Punnett square do.
    """
    return list(itertools.product(*(trait["parents"][parent] for trait in traits)))


def predict_cross(cross):
    """Build the Punnett square and the genotype/phenotype probabilities for a cross.

    Traits are assumed to assort independently (Mendel's second law). Probabilities are
    exact `Fraction`s, most likely first.
    """
    traits = cross["traits"]
    rows, columns = gametes(traits, 0), gametes(traits, 1)

    cells = []
    genotype_counts, phenotype_counts = Counter(), Counter()
    for row in rows:
        cell_row = []
        for column in columns:
            genotype = [
                _normalise(trait, (a, b)) for trait, a, b in zip(traits, row, column)
            ]
            label = " ".join(_label(g) for g in genotype)
            cell_row.append(label)
            genotype_counts[label] += 1
            phenotype_counts[
                ", ".join(expressed_phenotype(t, g) for t, g in zip(traits, genotype))
            ] += 1
        cells.append(cell_row)

    total = len(rows) * len(columns)
    return {
        "traits": [trait["name"] for trait in traits],
        "parents": [" ".join(_label(t["parents"][i]) for t in traits) for i in (0, 1)],
        "assumptions": cross["assumptions"],
        "punnett_square": {
            "rows": [_label(g) for g in rows],
            "columns": [_label(g) for g in columns],
            "cells": cells,
        },
        "genotypes": _probabilities(genotype_counts, total),
        "phenotypes": _probabilities(phenotype_counts, total),
    }


def _probabilities(counts, total):
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {outcome: Fraction(count, total) for outcome, count in ordered}


def _ratio(probabilities):
    denominator = math.lcm(*(p.denominator for p in probabilities.values()))
    parts = [int(p * denominator) for p in probabilities.values()]
    divisor = math.gcd(*parts)
    return ":".join(str(part // divisor) for part in parts)


def format_prediction(result):
    """Render a `predict_cross` result as text: the Punnett square, then both tables."""
    square = result["punnett_square"]
    header = [""] + square["columns"]
    body = [[label] + cells for label, cells in zip(square["rows"], square["cells"])]
    widths = [max(len(row[i]) for row in [header] + body) for i in range(len(header))]

    def line(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()

    lines = [f"Cross: {result['parents'][0]} x {result['parents'][1]} ({', '.join(result['traits'])})"]
    for assumption in result["assumptions"]:
        lines.append(f"Assumption: {assumption}")
    if len(result["traits"]) > 1:
        lines.append("Traits are assumed to assort independently (unlinked genes).")
    lines += ["", "Punnett square (rows: parent 1, columns: parent 2)", line(header)]
    lines.append("-+-".join("-" * width for width in widths))
    lines += [line(row) for row in body]

    for title, key in (("Genotypes", "genotypes"), ("Phenotypes", "phenotypes")):
        probabilities = result[key]
        lines += ["", f"{title} (ratio {_ratio(probabilities)})"]
        for outcome, p in probabilities.items():
            lines.append(f"  {outcome}: {p} ({round(float(p) * 100, 2):g}%)")
    return "\n".join(lines)
