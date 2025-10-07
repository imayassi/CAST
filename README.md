# Capacity Analytics & Simulation Toolkit (CAST)

**CAST** is a Python framework for analyzing team capacity, workload stress, and optimal headcount distribution.  
It helps operations and business leaders understand how teams are performing, identify imbalances, and simulate "what-if" staffing scenarios — all powered by interpretable analytics and machine learning.

---

## 🚀 What CAST Does

- **Capacity Score (0–100)** – Quantifies each employee’s operational output relative to their peers using a machine-learning model.  
- **Workload Stress Index (WSI, 0–100)** – Measures workload pressure by blending capacity deviation, persistence, complexity, and time pressure.  
- **Driver Analysis** – Explains what factors drive high or low Capacity Scores at both individual and organizational levels.  
- **Scenario Simulation** – Models the impact of moving engineers between teams on key metrics like WSI, efficiency, and backlog.  
- **Optimization Engine** – Recommends optimal headcount distribution to maximize efficiency and volume while minimizing stress and backlog.  
- **Automated Executive Summary** – Generates an HTML report with charts and LLM-powered narratives for non-technical audiences.

---

## 🧩 Core Concepts

### 1. Capacity Score
A 0–100 measure of **operational capacity and performance** relative to team benchmarks.  
Built using a transparent machine-learning model (**DSLI – Directional Sparse Logistic Index**) that evaluates:

| Dimension | Example Features |
|------------|------------------|
| Workload   | `numCases`, `backlog`, `TimeSpent` |
| Efficiency | `avgDaysToClose`, `som`, `activeTimeRatio` |
| Experience | `tenure`, `AICS`, `LinearityScore` |

Higher Capacity = greater ability to handle volume efficiently.

---

### 2. Workload Stress Index (WSI)
A 0–100 composite index that quantifies **pressure and sustainability** of workload.  
It combines four interpretable factors:

| Component | Description |
|------------|-------------|
| **Capacity Deviation (45%)** | How current load compares to an engineer’s typical baseline. |
| **Persistence (25%)** | Duration of sustained high workload. |
| **Complexity (15%)** | Case difficulty and severity mix. |
| **Time Pressure (15%)** | Weekend/after-hours or 24×7 commitments. |

Higher WSI = higher stress risk; teams target **high Capacity + moderate WSI**.

---

## ⚙️ How to Use CAST

### 1. Installation
```bash
pip install pandas numpy scikit-learn seaborn matplotlib openai markdown
