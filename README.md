# neumann

A small local engine that uses a quantized Qwen2.5-3B model (via llama-cpp-python) to power two tools:

- **genetics** (`engine/genetics/genetics.py`): predicts genetic inheritance patterns with Punnett squares.
- **ecology** (`engine/ecology/ecology.py`): estimates ecological figures for places like countries, cities, and oceans.

## Setup

```
pip install -r requirements.txt
python setup.py
```

`setup.py` downloads the model into `engine/models/`.


