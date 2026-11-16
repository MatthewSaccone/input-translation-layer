# CS4470 Input Translation Layer

## Project Overview

An LLM-assisted input translation layer for a unified
computational drug discovery platform.

The system translates natural-language researcher requests
into validated inputs for different scientific algorithms.

## Supported Algorithms

- MAC-AMP: Antimicrobial peptide design
- RL-Diffusion: Molecular generation and optimization

## Proposed Architecture

User Prompt
    |
    v
LLM Interpreter
    |
    v
Schema Validation
    |
    v
Algorithm-Specific Adapter
    |
    v
Prepared Algorithm Input

## Team Responsibilities

1. LLM Interpreter and AI Integration
2. Core Schemas, Registry, and Validation
3. MAC-AMP Input Adapter
4. RL-Diffusion Input Adapter

## Initial Scope

Our first milestone focuses on translating user prompts
into validated, algorithm-specific input specifications.

Full model execution and HU LAB frontend integration
will be addressed separately.
