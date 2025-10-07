# Capacity Analytics & Simulation Toolkit (CAST)

This repository contains the **Capacity Analytics & Simulation Toolkit (CAST)**, a Python-based framework for analyzing team capacity, workload stress, and optimizing headcount distribution. Using operational data, it calculates key performance indicators, simulates "what-if" scenarios, and provides data-driven recommendations for improving team efficiency and balance. The final output is a comprehensive, self-contained HTML executive summary generated with the help of an LLM.

---

## Key Features

-   **Capacity Score:** Calculates a 0-100 machine learning-based score to measure each employee's performance against their team's benchmark.
-   **Workload Stress Index (WSI):** Computes a 0-100 composite index to quantify workload pressure, blending capacity deviation, persistence, complexity, and time pressure.
-   **Driver Analysis:** Identifies the most influential factors driving the Capacity Score across the organization and for individual employees.
-   **"What-If" Scenarios:** A simulation engine to project the impact of moving employees between teams on key metrics like WSI, efficiency, and backlog.
-   **Optimization Engine:** Provides an optimal headcount reassignment plan to maximize business objectives (e.g., reduce WSI and backlog).
-   **Automated Reporting:** Generates a text-based executive summary and a complete HTML report with embedded visualizations using an LLM.

---

## The Core CAST Framework

The **CAST** framework is built around two primary metrics which serve as the foundation for all subsequent analyses.

### 1. The Capacity Score
The Capacity Score is not a simple metric. It's calculated using a machine learning model (**DSLI - Dissimilarity-based Sparse Linear Model**) that learns the patterns associated with high-capacity performance from your historical data.

-   **How it works:** It compares each employee to a "benchmark" performer on their team across various features (e.g., `numCases`, `backlog`, `TimeSpent`).
-   **Output:** A 0-100 score where a higher score indicates greater available capacity to handle work.

### 2. The Workload Stress Index (WSI)
The WSI is a transparent, rule-based composite index designed to measure the overall pressure an employee is under. It is a weighted average of four key components:

1.  **Capacity Deviation (40% weight):** How an employee's current capacity score deviates from their historical baseline. A negative deviation increases stress.
2.  **Persistence (20% weight):** Measures how long an employee has sustained a high capacity score, indicating prolonged periods of high output.
3.  **Complexity (20% weight):** The difficulty of the cases being handled, typically based on severity levels.
4.  **Time Pressure (20% weight):** Factors in schedule constraints, such as weekend work or 24x7 support obligations.

---

## How to Use CAST

### 1. Installation
This project uses several Python libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib openai markdown
