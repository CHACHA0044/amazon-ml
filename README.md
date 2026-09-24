# Amazon ML Challenge 2026

This repository contains the team's code, experiments, documentation, and reproducible pipeline for the Amazon ML Challenge 2026 entity-resolution problem.

## Project Structure

    dataset/
        Local competition dataset - NOT tracked in Git

    src/
        Main source code

    notebooks/
        Experiments and analysis

    scripts/
        Utility/training/evaluation scripts

    output/
        Generated outputs - only commit appropriate lightweight artifacts

    docs/
        Documentation and approach notes

## Dataset

The competition dataset is intentionally kept local and is excluded from Git because of its size and competition/data-sharing considerations. It is not part of this repository; configuring the dataset path is required to run the pipeline (see Reproducibility).

## Goal

The project will build a high-recall candidate-generation/blocking pipeline followed by entity matching and conservative set-level prediction for the Amazon ML Challenge.

## Reproducibility

Code should be designed so that another team member can run the pipeline against the locally available competition dataset after configuring the dataset path.