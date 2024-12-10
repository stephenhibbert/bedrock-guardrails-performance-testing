# Guardrail Evaluation Framework

This project provides a framework for evaluating Amazon Bedrock guardrails against prompt attack datasets. It includes automated dataset collection, evaluation metrics calculation, and visualization generation.

## Example Outputs

`make eval`
```
Evaluating LLM as judge...
LLM Judge: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [05:09<00:00,  2.07s/examples]
Evaluating Meta Prompt-Guard...
Device set to use mps:0
Meta Prompt-Guard: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [00:17<00:00,  8.40examples/s]
Evaluating AWS Bedrock...
AWS Bedrock: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [00:45<00:00,  3.30examples/s]
==================================================
PROMPT ATTACK DETECTION EVALUATION REPORT
==================================================

Total examples evaluated: 150

Meta Prompt-Guard Performance:
  Accuracy: 84.00%
  Precision: 100.00%
  Recall: 73.63%
  F1 Score: 84.81%
  False Positive Rate: 0.00%
  False Negative Rate: 16.00%

  Detailed Metrics:
    Class 0 (Benign):
      Precision: 71.08%
      Recall: 100.00%
    Class 1 (Malicious):
      Precision: 100.00%
      Recall: 73.63%

AWS Bedrock Performance:
  Accuracy: 72.00%
  Precision: 98.04%
  Recall: 54.95%
  F1 Score: 70.42%
  False Positive Rate: 0.67%
  False Negative Rate: 27.33%

  Detailed Metrics:
    Class 0 (Benign):
      Precision: 58.59%
      Recall: 98.31%
    Class 1 (Malicious):
      Precision: 98.04%
      Recall: 54.95%

LLM Judge Performance:
  Accuracy: 72.67%
  Precision: 96.30%
  Recall: 57.14%
  F1 Score: 71.72%
  False Positive Rate: 1.33%
  False Negative Rate: 26.00%

  Detailed Metrics:
    Class 0 (Benign):
      Precision: 59.38%
      Recall: 96.61%
    Class 1 (Malicious):
      Precision: 96.30%
      Recall: 57.14%
==================================================
```

`make data`

```
category     label
SPML         0         9
             1        41
bitext       0        50
hackaprompt  1        50
```

## Prerequisites

- Python 3.8+
- AWS credentials configured with access to Amazon Bedrock
- Terraform installed
- Required Python packages

## Project Structure

```
.
├── Makefile                    # Build automation
├── eval.py                     # Main evaluation script
├── get_datasets.py            # Dataset preparation script
├── terraform/                 # Terraform configurations
└── prompt_attack_dataset.csv  # Generated dataset file 
```

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure AWS credentials with appropriate permissions for Bedrock access
4. Ensure Terraform is installed and available in your PATH

## Usage

The project includes a Makefile with the following commands:

```bash
# Fetch and prepare datasets
make data

# Run evaluation
make eval

# Run both dataset preparation and evaluation
make all

# Clean generated files
make clean
```

## Evaluation Process

1. The framework first creates an evaluation directory with timestamp
2. Loads the prompt attack dataset
3. Deploys required AWS resources using Terraform
4. Runs evaluations
5. Generates metrics
6. Cleans up AWS resources

## Output

The evaluation generates the following in a timestamped directory (`eval_run_YYYYMMDD_HHMMSS/`):

## Error Handling

- The framework includes automatic cleanup of AWS resources even if evaluation fails
- Rate limiting is implemented to prevent API throttling

## Contributing

Please ensure all code changes include appropriate error handling and follow the existing logging patterns. Update tests as needed for new functionality.