# -*- coding: utf-8 -*-
"""
Subpacote de ingestão de dados (Módulo 1).
Reexporta a API pública estável — os nomes abaixo existem todos em
nu_data_pipeline.py, pelo que este import não pode falhar por name drift.
"""
from .nu_data_pipeline import (
    run_pipeline,
    build_nu_dataset,
    Dataset,
    NuDataset,
    validate_schema,
)

__all__ = [
    "run_pipeline",
    "build_nu_dataset",
    "Dataset",
    "NuDataset",
    "validate_schema",
]