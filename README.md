# Knowledge-Grounded Explainable Failure-Mode Diagnosis from Sensor-State Graphs

This directory contains the public code release for the paper "Knowledge-Grounded Explainable Failure-Mode Diagnosis from Sensor-State Graphs".

## Contents

The release contains two runnable components.

1. AI4I benchmark reformulation assets under AI4I_reformulation
2. A single-case DC-GBRF diagnosis notebook under DC-GBRF

## Entry Points

- AI4I benchmark assets: AI4I_reformulation
- Main diagnosis notebook: DC-GBRF/notebooks/llm_diagnosis.ipynb

The diagnosis notebook uses benchmark artifacts already stored inside PublicCode, so the diagnosis path runs entirely within this directory.

## Dataset

The benchmark reformulation workflow is based on the AI4I 2020 Predictive Maintenance Dataset.

- UCI dataset page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Expected local CSV path: AI4I_reformulation/dataset/ai4i2020.csv

## Setup

Create and activate a Python environment, install the required packages, and set Azure OpenAI environment variables before running the notebook.

Required environment variables for the diagnosis notebook:

- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION_CHAT
- AZURE_OPENAI_API_VERSION_EMBEDDINGS
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT

A template is provided in DC-GBRF/.env.example.

## Running the Diagnosis Notebook

Open DC-GBRF/notebooks/llm_diagnosis.ipynb and run the cells in order.

- The early cells validate local assets and build the retrieval context without making an API call.
- The final cell calls Azure OpenAI and prints three sections: Observed Evidence Summary, Candidate Context, and Raw LLM Output.

If SAVE_OUTPUT is set to True in the notebook, the final cell writes the raw LLM response to DC-GBRF/outputs/single_case_runs as a markdown file.

## Repository Structure

```text
PublicCode/
├── README.md
├── requirements.txt
├── AI4I_reformulation/
└── DC-GBRF/
    ├── .env.example
    ├── assets/
    ├── notebooks/
    ├── outputs/
    └── src/
```

## Security

This release does not store API keys, private endpoints, or private local paths in tracked files.

Keep all credentials in local environment variables or local ignored files outside the repository.
