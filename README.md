# neumann-engine

Predict the offspring of a genetic cross from plain English. A llama.cpp model
parses your description; everything else (Punnett square, genotype and
phenotype probabilities) is ordinary code with exact fractions.

```python
from genetics import load_model, predict, format_prediction

llm = load_model("qwen2.5-3b-instruct-q4_k_m.gguf")  # any GGUF instruct model
result = predict(llm, "Cross a heterozygous tall pea plant with a short one.")
print(format_prediction(result))
```

```
Cross: Tt x tt (height)

Punnett square (rows: parent 1, columns: parent 2)
  | t  | t
--+----+---
T | Tt | Tt
t | tt | tt

Genotypes (ratio 1:1)
  Tt: 1/2 (50%)
  tt: 1/2 (50%)

Phenotypes (ratio 1:1)
  short: 1/2 (50%)
  tall: 1/2 (50%)
```

## Functions

| Function | What it does |
| --- | --- |
| `load_model(path)` | Load a GGUF model with `llama-cpp-python`. |
| `parse_input(llm, text)` | Model turns the description into JSON (grammar-constrained). |
| `build_cross(parsed)` | Validate the parsed cross. Raises `ValueError` if it makes no genetic sense. |
| `predict_cross(cross)` | Punnett square plus genotype/phenotype probabilities (`Fraction`s). |
| `predict(llm, text)` | The three above in one call, retrying if the model's answer is rejected. |
| `format_prediction(result)` | Render a result as text. |

`build_cross` and `predict_cross` need no model, so you can pass them a
hand-written cross (see `test_genetics.py`).

## Supported

- Complete dominance, incomplete dominance, codominance
- Multiple alleles (e.g. ABO blood type)
- One X-linked trait (results are split by sex)
- Up to 4 traits at once, assumed to assort independently (unlinked genes)

When the description leaves a parent's genotype open, the model is asked to
say what it assumed; those notes appear in `result["assumptions"]`.

## Setup

```
pip install -r requirements.txt
pip install pytest && pytest
```

The tests use a stand-in for the model, so they run without a GGUF file.
