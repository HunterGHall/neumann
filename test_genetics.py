from fractions import Fraction
from types import SimpleNamespace

import pytest

import genetics


def trait(name, alleles, parent1, parent2, x_linked=False, blend=""):
    """Shorthand for one parsed trait; `alleles` is (symbol, phenotype, dominance) tuples."""
    return {
        "name": name,
        "x_linked": x_linked,
        "alleles": [genetics._allele(*allele) for allele in alleles],
        "heterozygous_phenotype": blend,
        "parent1": parent1,
        "parent2": parent2,
    }


def cross(*traits, assumptions=()):
    return genetics.predict_cross(
        genetics.build_cross({"traits": list(traits), "assumptions": list(assumptions)})
    )


def height(parent1, parent2):
    return trait(
        "height", [("T", "tall", "dominant"), ("t", "short", "recessive")], parent1, parent2
    )


def fake_llm(*replies):
    """A stand-in for `Llama` that returns each reply (a JSON string or dict) in turn."""
    calls = []

    def create_chat_completion(**kwargs):
        calls.append(kwargs)
        reply = replies[min(len(calls), len(replies)) - 1]
        content = reply if isinstance(reply, str) else genetics.json.dumps(reply)
        return {"choices": [{"message": {"content": content}}]}

    return SimpleNamespace(create_chat_completion=create_chat_completion, calls=calls)


# ------------------------------------------------------------------ punnett squares


def test_monohybrid_square_and_probabilities():
    result = cross(height(["T", "t"], ["T", "t"]))
    assert result["punnett_square"] == {
        "rows": ["T", "t"],
        "columns": ["T", "t"],
        "cells": [["TT", "Tt"], ["Tt", "tt"]],
    }
    assert result["genotypes"] == {
        "Tt": Fraction(1, 2), "TT": Fraction(1, 4), "tt": Fraction(1, 4),
    }
    assert result["phenotypes"] == {"tall": Fraction(3, 4), "short": Fraction(1, 4)}


def test_homozygous_parent_repeats_its_gamete():
    result = cross(height(["T", "T"], ["T", "t"]))
    assert result["punnett_square"]["rows"] == ["T", "T"]
    assert result["genotypes"] == {"TT": Fraction(1, 2), "Tt": Fraction(1, 2)}
    assert result["phenotypes"] == {"tall": Fraction(1)}


def test_allele_order_in_input_does_not_matter():
    assert cross(height(["t", "T"], ["t", "t"])) == cross(height(["T", "t"], ["t", "t"]))


def test_test_cross_is_one_to_one():
    result = cross(height(["T", "t"], ["t", "t"]))
    assert result["phenotypes"] == {"tall": Fraction(1, 2), "short": Fraction(1, 2)}
    assert genetics._ratio(result["phenotypes"]) == "1:1"


def test_dihybrid_cross_gives_nine_three_three_one():
    result = cross(
        trait("shape", [("R", "round", "dominant"), ("r", "wrinkled", "recessive")], ["R", "r"], ["R", "r"]),
        trait("color", [("Y", "yellow", "dominant"), ("y", "green", "recessive")], ["Y", "y"], ["Y", "y"]),
    )
    square = result["punnett_square"]
    assert square["rows"] == ["RY", "Ry", "rY", "ry"]
    assert len(square["cells"]) == 4 and all(len(row) == 4 for row in square["cells"])
    assert list(result["phenotypes"].items()) == [
        ("round, yellow", Fraction(9, 16)),
        ("round, green", Fraction(3, 16)),
        ("wrinkled, yellow", Fraction(3, 16)),
        ("wrinkled, green", Fraction(1, 16)),
    ]
    assert genetics._ratio(result["phenotypes"]) == "9:3:3:1"
    assert sum(result["genotypes"].values()) == 1 and len(result["genotypes"]) == 9


# ------------------------------------------------------------------ other inheritance


def test_incomplete_dominance():
    result = cross(
        trait(
            "flower color", [("R", "red", "codominant"), ("W", "white", "codominant")],
            ["R", "W"], ["R", "W"], blend="pink",
        )
    )
    assert result["phenotypes"] == {
        "pink": Fraction(1, 2), "red": Fraction(1, 4), "white": Fraction(1, 4),
    }


def test_codominance_shows_both_phenotypes():
    result = cross(
        trait(
            "coat", [("R", "red", "codominant"), ("W", "white", "codominant")],
            ["R", "R"], ["W", "W"],
        )
    )
    assert result["phenotypes"] == {"red and white": Fraction(1)}


def test_abo_blood_type_multiple_alleles():
    result = cross(
        trait(
            "blood type",
            [("IA", "A", "codominant"), ("IB", "B", "codominant"), ("i", "O", "recessive")],
            ["IA", "i"], ["IB", "i"],
        )
    )
    assert result["parents"] == ["IA/i", "IB/i"]
    assert set(result["genotypes"]) == {"IA/IB", "IA/i", "IB/i", "ii"}
    assert result["phenotypes"] == {
        "A and B": Fraction(1, 4), "A": Fraction(1, 4), "B": Fraction(1, 4), "O": Fraction(1, 4),
    }


def test_x_linked_carrier_mother_and_colorblind_father():
    result = cross(
        trait(
            "color vision",
            [("XB", "normal", "dominant"), ("Xb", "colorblind", "recessive")],
            ["Xb", "Y"], ["XB", "Xb"], x_linked=True,
        )
    )
    assert result["parents"] == ["Xb/Y", "XB/Xb"]
    assert result["phenotypes"] == {
        "colorblind (female)": Fraction(1, 4),
        "colorblind (male)": Fraction(1, 4),
        "normal (female)": Fraction(1, 4),
        "normal (male)": Fraction(1, 4),
    }


def test_x_linked_trait_with_autosomal_trait_keeps_sex_consistent():
    result = cross(
        trait("height", [("T", "tall", "dominant"), ("t", "short", "recessive")], ["T", "T"], ["t", "t"]),
        trait(
            "color vision", [("XB", "normal", "dominant"), ("Xb", "colorblind", "recessive")],
            ["XB", "Y"], ["XB", "Xb"], x_linked=True,
        ),
    )
    assert result["phenotypes"] == {
        "tall, normal (female)": Fraction(1, 2),
        "tall, normal (male)": Fraction(1, 4),
        "tall, colorblind (male)": Fraction(1, 4),
    }


# ------------------------------------------------------------------ validation


def parsed(*traits):
    return {"traits": list(traits), "assumptions": []}


def test_unknown_allele_is_rejected():
    with pytest.raises(ValueError, match="must be two of the alleles"):
        genetics.build_cross(parsed(height(["T", "x"], ["t", "t"])))


def test_duplicate_allele_symbol_is_rejected():
    bad = trait("height", [("T", "tall", "dominant"), ("T", "short", "recessive")], ["T", "T"], ["T", "T"])
    with pytest.raises(ValueError, match="listed twice"):
        genetics.build_cross(parsed(bad))


def test_too_many_traits_is_rejected():
    with pytest.raises(ValueError, match="between 1 and"):
        genetics.build_cross(parsed(*[height(["T", "t"], ["T", "t"])] * (genetics.MAX_TRAITS + 1)))


def x_linked(parent1, parent2):
    return trait(
        "color vision", [("XB", "normal", "dominant"), ("Xb", "colorblind", "recessive")],
        parent1, parent2, x_linked=True,
    )


def test_two_males_cannot_be_crossed():
    with pytest.raises(ValueError, match="both parents carry Y"):
        genetics.build_cross(parsed(x_linked(["XB", "Y"], ["Xb", "Y"])))


def test_y_cannot_be_listed_as_an_x_linked_allele():
    bad = trait("vision", [("Y", "a", "dominant"), ("Xb", "b", "recessive")], ["Xb", "Xb"], ["Xb", "Xb"], x_linked=True)
    with pytest.raises(ValueError, match="do not list Y"):
        genetics.build_cross(parsed(bad))


def test_only_one_x_linked_trait():
    with pytest.raises(ValueError, match="only one X-linked"):
        genetics.build_cross(parsed(x_linked(["XB", "Y"], ["XB", "Xb"]), x_linked(["XB", "Y"], ["XB", "Xb"])))


def test_parent_with_two_y_chromosomes_is_rejected():
    with pytest.raises(ValueError, match="two Y"):
        genetics.build_cross(parsed(x_linked(["Y", "Y"], ["XB", "Xb"])))


# ------------------------------------------------------------------ text output


def test_format_prediction():
    text = genetics.format_prediction(cross(height(["T", "t"], ["T", "t"]), assumptions=["guessed"]))
    assert text == "\n".join([
        "Cross: Tt x Tt (height)",
        "Assumption: guessed",
        "",
        "Punnett square (rows: parent 1, columns: parent 2)",
        "  | T  | t",
        "--+----+---",
        "T | TT | Tt",
        "t | Tt | tt",
        "",
        "Genotypes (ratio 2:1:1)",
        "  Tt: 1/2 (50%)",
        "  TT: 1/4 (25%)",
        "  tt: 1/4 (25%)",
        "",
        "Phenotypes (ratio 3:1)",
        "  tall: 3/4 (75%)",
        "  short: 1/4 (25%)",
    ])


def test_ratio_of_a_certain_outcome_is_one():
    assert genetics._ratio({"tall": Fraction(1)}) == "1"


# ------------------------------------------------------------------ model plumbing


def test_prompt_examples_are_valid_crosses():
    for text, answer in genetics.EXAMPLES:
        result = genetics.predict_cross(genetics.build_cross(answer))
        assert sum(result["phenotypes"].values()) == 1, text


def test_parse_input_sends_schema_and_examples_then_returns_parsed_json():
    answer = genetics.EXAMPLES[0][1]
    llm = fake_llm(answer)
    assert genetics.parse_input(llm, "my cross") == answer

    (call,) = llm.calls
    assert call["response_format"] == {"type": "json_object", "schema": genetics.PARSE_SCHEMA}
    roles = [message["role"] for message in call["messages"]]
    assert roles == ["system"] + ["user", "assistant"] * len(genetics.EXAMPLES) + ["user"]
    assert call["messages"][-1]["content"] == "my cross"


def test_predict_end_to_end_with_model_reply():
    llm = fake_llm({"traits": [height(["T", "t"], ["t", "t"])], "assumptions": []})
    result = genetics.predict(llm, "Cross a heterozygous tall plant with a short one.")
    assert result["phenotypes"] == {"tall": Fraction(1, 2), "short": Fraction(1, 2)}


def test_predict_retries_with_the_error_when_the_reply_makes_no_sense():
    bad = {"traits": [height(["T", "x"], ["t", "t"])], "assumptions": []}
    good = {"traits": [height(["T", "t"], ["t", "t"])], "assumptions": []}
    llm = fake_llm(bad, "not json{", good)

    result = genetics.predict(llm, "a cross")
    assert result["parents"] == ["Tt", "tt"]
    assert len(llm.calls) == 3
    assert "must be two of the alleles" in llm.calls[1]["messages"][-1]["content"]
    assert llm.calls[0]["messages"][-1]["content"] == "a cross"


def test_predict_gives_up_after_the_attempts_are_used():
    llm = fake_llm({"traits": [height(["T", "x"], ["t", "t"])], "assumptions": []})
    with pytest.raises(ValueError, match="could not get a valid cross"):
        genetics.predict(llm, "a cross", attempts=2)
    assert len(llm.calls) == 2


def test_schema_compiles_to_a_llama_cpp_grammar():
    llama_grammar = pytest.importorskip("llama_cpp.llama_grammar")
    grammar = llama_grammar.LlamaGrammar.from_json_schema(genetics.json.dumps(genetics.PARSE_SCHEMA))
    assert grammar is not None
