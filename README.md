# RAG demo

## Introduction

The goal of this demo app is to showcase the quickest and easiest way to build a usable knowledge database using Retrieval-Augmented Generation (RAG). RAG enhances the ability to answer questions by combining retrieval and generation capabilities.

Use case: seed knowledge with maintenance manuals, ask questions for troubleshooting issues. Provided is an example PDF file (t60.pdf - maintenance manual for IBM T60 laptops).

## Features

- Retrieval-Augmented Generation implementation via LangChain
- Dockerized setup for easy running
- Integration with AWS Bedrock
- Usage of LangChain for language model operations

## Prerequisites

- Docker
- AWS credentials with Bedrock access (ask @ivan.petrushev, if uncertain)

## Installation

1. Clone the repository
2. Copy the `.env.example` file to `.env` and fill in the required values

## Usage

Build the Docker image:

```bash
docker build -t local-rag:latest image/
```

Run the Docker container:

```bash
docker run -it --env-file .env local-rag:latest python3 pdfloader.py
```

This will:
- read the provided example PDF file
- extract text from the PDF
- split text into chunks
- generate chunk embeddings (this takes some time: ~3 min with the `amazon.titan-embed-text-v2:0` model)
- load embeddings to a local ChromaDB vector store
- run a loop with query-retrieve-generate operations

## Results

Some example questions and generated answers can be found in the `results/` directory.

## Improvement points

Tons of them, but here are a few:

- compare different vector stores: https://python.langchain.com/v0.1/docs/integrations/vectorstores/
- compare different chunking strategies (different tokenizers, window size, overlap, etc)
- compare different language models for generating answers
- experiment with different LLM prompt instructions
- when comparing anything, we will need a result evaluation metric/strategy
- image extraction - a whole topic on its own
- streaming responses
- experiment with local Huggingface models to cut out Bedrock dependency
- add memory to context - currently each question is treated as a separate context
- experiment with different type of text information - source code? software manuals?
- improve retrieval process with query derivation from user input

## Costs

Using AWS Bedrock will incur some costs. 

With the current setup (example t60.pdf file of ~200 pages mixed text and images, settings for chunking text) the whole corpus is about 200 000 tokens. Running it through the `amazon.titan-embed-text-v2:0` model will cost about $0.004.

Asking questions and sending prompts to `anthropic.claude-3-haiku-20240307-v1:0` results in minimal costs for input/output model tokens. For rough estimate - everything in the `results/` direcory was generated for about $0.005.

Keep in mind those are one of the cheapest models, so probably a model switch will increase costs.