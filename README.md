# Phi-4 Chat Interface

Simple command-line interface for Microsoft's Phi-4 language model with quantization options.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run with default settings (4-bit quantization):
```bash
python run_phi4.py
```

Specify quantization level:
```bash
# 4-bit quantization
python run_phi4.py -q 4

# 8-bit quantization
python run_phi4.py -q 8

# No quantization
python run_phi4.py -q none
```

Type 'exit' or 'quit' to end the conversation. 