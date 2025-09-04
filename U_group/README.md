# Group Understanding Experiment

A multi-agent communication experiment framework for measuring group understanding dynamics using novel non-compensatory metrics.

Theoretical background and equations are detailed in the [associated paper](group.pdf).

## Overview

This tool simulates conversations between AI agents to study how understanding converges in multi-agent systems. It implements a Q&A protocol where agents ask clarifying questions about an initial query and measures understanding using Jensen-Shannon divergence and semantic similarity metrics.

## Features

**Multi-agent Q&A Protocol**: Agents take turns asking clarifying questions and updating their understanding

### Understanding Metrics

- Per-turn group understanding scores $U_{group}^{(n)}$
- Cumulative discussion-level understanding $U_{discussion}$
- Pairwise understanding matrices $u_{ij}$

**Early Stopping**: Automatic convergence detection based on variance and improvement thresholds

### Comprehensive Output

  - Detailed conversation logs
  - Understanding evolution tracking
  - Model-specific attribution
  - Rich visualizations

**Flexible Configuration**: Support for different AI models, local endpoints, and experimental parameters

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the project directory:

```bash
OPENAI_API_KEY="your_openai_api_key_here"
EMBEDDING_MODEL="text-embedding-3-large"
AGENTS="gpt-4.1-mini,gpt-4.1-mini,gpt-4.1-mini,gpt-4.1-mini"
LOCAL_AI_API_URL="http://localhost:8080/v1"
```

### Agent Configuration

Configure agents using the `AGENTS` environment variable:
- Comma-separated list of model names
- Each model becomes one agent in the experiment
- Example: `AGENTS="gpt-4,gpt-3.5-turbo,claude-3-sonnet"`

## Usage

### Basic Usage

Run without any additional parameters to use default settings. Use the `--help` flag to see all available options:

```bash
python group_experiment.py --help
usage: group_experiment.py [-h] [-m MAX_TURNS] [--min_turns MIN_TURNS] [--tau TAU] [-e EPSILON] [-t TEMPERATURE] [-l] [-k]
                           [--schedule {linear,uniform}] [--convergence_variance_threshold CONVERGENCE_VARIANCE_THRESHOLD]
                           [--convergence_improvement_threshold CONVERGENCE_IMPROVEMENT_THRESHOLD] [--topic_seed TOPIC_SEED]
                           [-q QUERY] [-g GT] [-i SPEAKER_INDEX] [-o OUTPUT_DIR] [--include_embeddings] [--no_plots] [--hyp_djs]
                           [-v]

Run multi-agent communication experiment with group understanding metrics

options:
  -h, --help            show this help message and exit
  -m, --max_turns MAX_TURNS
                        Maximum number of turns in the experiment (default: 8)
  --min_turns MIN_TURNS
                        Minimum number of turns before early stopping (default: 4)
  --tau TAU             Kappa retention temperature parameter (default: 1.0)
  -e, --epsilon EPSILON
                        Product floor to prevent hard zeros (default: 1e-06)
  -t, --temperature TEMPERATURE
                        LLM sampling temperature (default: 0.7)
  -l, --use_local       Use local AI endpoint instead of OpenAI (default: False)
  -k, --use_kappa       Enable kappa-based metrics (if implemented) (default: False)
  --schedule {linear,uniform}
                        Weighting schedule for discussion-level understanding (linear or uniform) (default: linear)
  --convergence_variance_threshold CONVERGENCE_VARIANCE_THRESHOLD
                        Variance threshold for early stopping convergence detection (default: 0.0001)
  --convergence_improvement_threshold CONVERGENCE_IMPROVEMENT_THRESHOLD
                        Average improvement threshold for early stopping convergence detection (default: 0.001)
  --topic_seed TOPIC_SEED
                        Topic seed for the discussion (Agent GT only) (default: Evaluating group consensus in multi-agent systems
                        with a non-compensatory metric.)
  -q, --query QUERY     Direct query for external GT (required with --gt) (default: None)
  -g, --gt GT           Ground truth statement for external query (required with --query) (default: None)
  -i, --speaker_index SPEAKER_INDEX
                        Index of the initial speaker agent (Agent GT only) (default: 0)
  -o, --output_dir OUTPUT_DIR
                        Output directory for results (default: creates timestamped folder) (default: None)
  --include_embeddings  Include embeddings in saved results (creates larger files) (default: False)
  --no_plots            Skip displaying plots (default: False)
  --hyp_djs             Use hypothesis-set based Jensen-Shannon divergence for cleaner information-fidelity signals (default:
                        False)
  -v, --verbose         Enable verbose output (default: False)
```

### Advanced Usage

Run with custom parameters:

```bash
python group_experiment.py \
    --max_turns 10 \
    --min_turns 3 \
    --topic_seed "Evaluating machine learning model performance" \
    --temperature 0.8 \
    --use_kappa \
    --verbose
```

Run with external query and ground truth statement:

```bash
python group_experiment.py \
    --use_kappa --temperature 0.2 --max_turns 20 \
    -q "What are the key principles of effective machine learning model evaluation?" \
    -g "Effective ML model evaluation requires comprehensive assessment across multiple dimensions: statistical performance metrics (accuracy, precision, recall, F1), robustness testing (adversarial examples, distribution shift), fairness evaluation (demographic parity, equalized opportunity), and practical considerations (computational efficiency, interpretability, deployment constraints)."
```

## Output Files

When you run an experiment, it creates a timestamped folder with:

### Data Files

- **`results.json`**: Complete experiment data including:
  - Conversation transcripts
  - Understanding statements
  - Metric scores
  - Agent and model information
  - Metadata

### Visualizations

- **`understanding_metrics.png`**: Dual subplot showing per-turn and cumulative scores
- **`combined_metrics.png`**: Combined view of both metrics
- **`pairwise_heatmap.png`**: Final understanding similarity matrix
- **`pairwise_evolution.png`**: Turn-by-turn evolution of pairwise similarities

## Example Workflows

### 1. Quick Test with Different Models

```bash
# Test with mixed models
AGENTS="gpt-4,gpt-3.5-turbo,got-4.1-mini" python group_experiment.py \
    --max_turns 6 \
    --verbose
```

### 2. Convergence Analysis

```bash
# Study convergence with sensitive thresholds
python group_experiment.py \
    --convergence_variance_threshold 0.00005 \
    --convergence_improvement_threshold 0.0005 \
    --min_turns 6 \
    --max_turns 15
```

### 3. Local Model Testing

*Note: local model is used for all agents, including the speaker. The AGENTS variable is ignored.*

```bash
# Use local AI endpoint
python group_experiment.py \
    --use_local \
    --temperature 0.5
```

### 4. Save Results to Specific Directory

```bash
python group_experiment.py \
    --output_dir "./experiment_results/test_run_1" \
    --include_embeddings
```

## Interpreting Results

### Understanding Scores

- **0.0 - 0.4**: Low understanding/high disagreement
- **0.4 - 0.7**: Moderate understanding/some alignment
- **0.7 - 1.0**: High understanding/strong alignment

### Convergence Indicators

- **Variance**: Low variance in recent scores indicates stability
- **Improvement**: Small improvements suggest diminishing returns
- **Early stopping**: Automatic termination when both criteria are met

### Pairwise Matrices

- **Diagonal**: Always 1.0 (self-similarity), it is not used for scoring
- **Off-diagonal**: Understanding between different agents
- **Evolution**: Shows how understanding develops over turns

## Advanced Configuration

### Custom Topic Seeds

Use domain-specific topics:

```bash
python group_experiment.py \
    --topic_seed "Implementing federated learning with differential privacy constraints"
```
