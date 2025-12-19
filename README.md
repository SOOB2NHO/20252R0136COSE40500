# 20252R0136COSE40500

Subset retrieval evaluation project for COSE40500.

## Description

This repository contains evaluation scripts for subset-based table retrieval performance analysis.

## Features

- Multiple subset evaluation (Baseline, Relevant, Best, LLM, QA subsets)
- Query-table and query-query similarity based retrieval
- Performance metrics: Recall@K, MRR
- Type-specific embedding support

## Usage

```bash
python evaluate.py --subset-dir <path> --sacu-dir <path>
```

