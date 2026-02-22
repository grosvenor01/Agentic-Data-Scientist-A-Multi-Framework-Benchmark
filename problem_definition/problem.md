# Problem Definition  
## Agentic Data Scientist – Multi-Framework Benchmark

---

## Overview

Modern agentic AI frameworks (LangGraph, CrewAI, AGNO, LlamaIndex, PydanticAI, etc.) offer different abstractions for building autonomous systems. However, developers lack practical, side-by-side comparisons demonstrating how these frameworks perform under identical conditions.

This project defines a **standardized benchmark problem**:

> **Design an autonomous AI system capable of performing end-to-end data science tasks on a dataset.**

The same task, tools, and evaluation criteria are applied across multiple frameworks to analyze trade-offs in:

- Latency  
- Reliability  
- Token usage / cost  
- Developer experience  
- Flexibility  
- Maintainability  

---

## Objective

Build and compare multiple implementations of an **Agentic Data Scientist** that can:

1. Load a dataset  
2. Perform Exploratory Data Analysis (EDA) **(using programatic tool calling)**
3. Clean / preprocess data  **(using programatic tool calling)**
4. Train machine learning models  **(using programatic tool calling)**
5. Evaluate model performance  
6. Generate a structured analytical report  

Each implementation uses a **different agentic AI framework**, while preserving:

- Identical task definition  
- Identical tools
- Identical dataset  
- Identical evaluation metrics  

---

## Core Task

Given a dataset, the AI system must autonomously:

### Data Understanding
- Inspect schema
- Identify data types
- Detect missing values
- Summarize statistics

### Data Preparation
- Handle missing values
- Encode categorical features
- Normalize / scale features (if needed)

### Modeling
- Select appropriate ML model(s)
- Train model(s)
- Justify choices

### Evaluation
- Compute performance metrics
- Interpret results

### Reporting
- Generate a structured explanation including:
  - Key insights
  - Data issues
  - Model performance
  - Limitations

---

## Agentic Architecture

The system is composed of specialized agents:

| Agent | Responsibility |
|------|----------------|
| **Planner** | Break task into steps |
| **Data Loader** | Load and validate dataset |
| **EDA Analyst** | Perform exploratory analysis |
| **Preprocessing Agent** | Clean / transform data |
| **Model Trainer** | Train ML models |
| **Evaluator** | Compute and interpret metrics |
| **Reporter** | Produce final report |

Framework implementations may vary in orchestration strategy but must maintain equivalent logical roles.

---

## Benchmark Goals

This project evaluates frameworks based on:

### Performance
- End-to-end latency
- Tool execution overhead

### Efficiency
- Token consumption
- Cost estimation

### Reliability
- Error rate
- Recovery behavior

### Developer Experience
- Code complexity
- Boilerplate required
- Debugging clarity

### Flexibility
- Ease of adding tools/agents

### Maintainability
- Readability
- Modularity
- Scalability

---

## Dataset Selection

Benchmark datasets should be:

- Publicly available
- Structured (CSV / tabular)
- Suitable for ML tasks

Examples:

- Titanic dataset  
- Iris dataset  
- Housing price dataset  

---

## Expected Output

Each framework version must produce:

1. Structured analysis report  
2. Model evaluation metrics  
3. Logs / traces (if supported)  
4. Benchmark measurements  

---

## Experimental Methodology

All implementations must:

- Use the same dataset
- Use the same ML library (e.g., scikit-learn)
- Use identical evaluation criteria
- Run under similar hardware conditions

---

## Success Criteria

A framework implementation is considered successful if it:

✔ Completes full pipeline autonomously  
✔ Produces coherent analysis  
✔ Executes tools correctly  
✔ Handles errors gracefully  

---

## Research Value

This project provides:

- Practical comparison of agentic AI frameworks  
- Insights into orchestration trade-offs  
- Guidance for framework selection  
- Reproducible evaluation setup  

---

## Future Extensions
 
- Multi-modal data (for now we will use csv only data)

---

