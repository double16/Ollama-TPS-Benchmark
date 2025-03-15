# Ollama TPS Benchmark

A Python utility to measure tokens per second (TPS) for Ollama language models. This tool allows you to benchmark the generation speed of different models with customizable prompts.

## Features

- Measures tokens per second (TPS) for Ollama models
- Supports both streaming and non-streaming modes
- Calculates time to first token for responsiveness measurement
- Handles different Ollama model families and API response formats
- Configurable prompt and API endpoint
- Option to export results to JSON for further analysis

## Requirements

- Python 3.6+
- Ollama running locally or on a network-accessible server
- Python `requests` library

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ollama-tps-benchmark.git
   cd ollama-tps-benchmark
   ```

2. Install dependencies:
   ```
   pip install requests
   ```

## Usage

Basic usage:

```bash
python benchmark.py --model llama2
```

This will run a benchmark using the default prompt "Tell me a joke" on the Llama2 model.

### Command Line Arguments

| Argument | Shorthand | Description | Default |
|----------|-----------|-------------|---------|
| `--model` | `-m` | Name of the Ollama model (required) | N/A |
| `--prompt` | `-p` | Prompt to send to the model | "Tell me a joke" |
| `--url` | `-u` | Base URL for the Ollama API | "http://localhost:11434" |
| `--stream` | `-s` | Use streaming mode for more accurate token counting | False |
| `--output` | `-o` | Output file for JSON results (optional) | N/A |

### Examples

Benchmark with a custom prompt:
```bash
python benchmark.py --model mistral --prompt "Explain quantum computing"
```

Use streaming mode for more accurate measurements:
```bash
python benchmark.py --model gemma3:4b --stream
```

Connect to a remote Ollama instance:
```bash
python benchmark.py --model llama3 --url http://192.168.1.100:11434
```

Save results to a JSON file:
```bash
python benchmark.py --model llama2 --output results.json
```

## Sample Output

```
OLLAMA BENCHMARK RESULTS
==================================================
Model: gemma3:4b
Prompt tokens: 4
Completion tokens: 73
Total tokens: 77
Time to first token: 0.6523 seconds
Processing time: 6.2045 seconds
Wall clock time: 6.2175 seconds
Tokens per second: 13.18
==================================================

Generated response:
--------------------------------------------------
Why did the scarecrow win an award? 
…Because he was outstanding in his field! 😂 
---
Would you like to hear another one?
--------------------------------------------------
```

## How It Works

The benchmark works in two modes:

1. **Non-streaming mode**: Sends the entire request at once and measures the total time to generate the response. Token counts are estimated if not provided by the API.

2. **Streaming mode** (recommended): Counts tokens as they arrive in real-time, providing more accurate measurements. This also allows measurement of time to first token.

The script calculates:
- Prompt tokens (input)
- Completion tokens (output)
- Total tokens
- Processing time (model generation time)
- Wall clock time (total request time)
- Time to first token (latency)
- Tokens per second (speed)

## Why Use This Tool

- Compare performance across different Ollama models
- Optimize your model selection for specific use cases
- Benchmark hardware configurations for Ollama deployments
- Test network performance with remote Ollama instances

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.