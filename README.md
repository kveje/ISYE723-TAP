# ISYE723-TAP

**Final Project: Team Assignment Problem**

## Introduction

This project addresses the Multi Period Team Assignment Problem (TAP), where individuals with evolving and partially observable preferences are assigned to teams over multiple periods. The framework uses a Kalman Filter-based learner and strategic team assignment methods, such as Upper Confidence Bound (UCB) and Thompson Sampling, to optimize performance and adapt to dynamic changes.

## Features

- Dynamic preference estimation using a Kalman Filter.
- Team assignment strategies:
  - **Random Assignment**: Baseline method.
  - **UCB**: Balances exploration and exploitation.
  - **Thompson Sampling**: Bayesian sampling for team assignments.
- Integer Programming (IP) for optimal team configurations.
- Simulation support for performance evaluation with varying system parameters.
- Mechanism to handle dynamic resets (e.g., hiring/firing).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kveje/ISYE723-TAP.git
   cd ISYE723-TAP
   ```
2. Create a virtual environment
   ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run installation with
```bash
python main.py
```
- Configure parameters like number of individuals, team size, and reset probability in main.py.
- 	View results in the results/ directory, including plots for rewards, preference distances, and computational times.

If plots are not directly created, run
```bash
python show.py
```
with the appropriate PATH in the file.