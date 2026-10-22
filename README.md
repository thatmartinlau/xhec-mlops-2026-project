<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

This **group 7**'s submission for the X-HEC MLOps course with Artefact.

Group members:
- Martin Lau (martin.lau@hec.edu)
- Yuqi Xin (yuqi.xin@hec.edu)
- Julian Fix (julian.fix@hec.edu)
- Scott Mayette (scott.mayette@hec.edu)

This project includes:
- Notebook exploration (EDA and a first model tracked with MLflow)
- A training pipeline orchestrated with Prefect (`src/modelling`)
- A Prefect deployment that retrains the model every week


## 🚀 Quick Start

### Setup Steps

1. **Set up your development environment**:
   ```bash
   # Create and activate a virtual environment
   uv sync
   source .venv/bin/activate # on Windows (Git Bash): source .venv/Scripts/activate

   # Install pre-commit hooks for code quality
    uv pip install pre-commit
    uv run pre-commit install
   ```

2. **Download the data**: get `abalone.csv` from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and save it as `data/abalone.csv` (the `data/` folder is git-ignored).

### Training pipeline with Prefect

The training code in `src/modelling` is a Prefect **flow** (`Training pipeline`, in `main.py`) made of **tasks**, one per step: load data, remove invalid rows, construct the target, extract X and y, split train/test, build the preprocessor, build the pipeline, train (fits the preprocessor and the model together), evaluate and save the model. The trained pipeline is saved to `src/web_service/local_objects/model.pkl`.

All commands below are run from the project root.

1. **Point Prefect to a local server** (only needed once):
   ```bash
   uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   ```

2. **Start the Prefect server and UI** in a first terminal (it keeps running):
   ```bash
   uv run prefect server start
   ```
   Open the UI at http://127.0.0.1:4200. To reset the database: `uv run prefect server database reset`.

3. **Run the training flow once** in a second terminal:
   ```bash
   uv run python -m src.modelling.main data/abalone.csv
   ```
   The run appears in the UI under **Runs**: click on it to see each task, its logs and the test metrics (RMSE, MAE, R²).

4. **Deploy the weekly retraining** in the second terminal (it keeps running to serve the scheduled runs):
   ```bash
   uv run python -m src.modelling.deployment
   ```
   The `weekly-retraining` deployment appears in the UI under **Deployments**, scheduled every Monday at 00:00 (cron `0 0 * * 1`, set in `src/modelling/deployment.py`). To trigger a run now, click **Run** in the UI or use, from a third terminal:
   ```bash
   uv run prefect deployment run "Training pipeline/weekly-retraining"
   ```
   Stopping the command (`Ctrl+C`) stops the scheduled runs; the deployment stays visible in the UI.

## 📋 What You'll Build

By the end of this project, you'll have created:

### 🤖 **Automated ML Pipeline**
- Training workflows using Prefect
- Automatic model retraining on schedule
- Reproducible model and data processing

### 🌐 **Prediction API**
- REST API for real-time predictions
- Input validation with Pydantic
- Docker containerization

### 📊 **Production-Ready Code**
- Clean, well-documented code
- Automated testing and formatting
- Proper error handling

## 📝 How to Work on This Project

### The Branch-by-Branch Approach

This project is organized into numbered branches, each representing a step in building your MLOps system. Think of it like a guided tutorial where each branch teaches you something new!

**Here's how it works**:

1. **Each branch = One pull request** with specific tasks
2. **Follow the numbers** (branch_0, branch_1, etc.) in order
3. **Read the PR instructions** (PR_0.md, PR_1.md, etc.) before starting
4. **Complete all TODOs** in that branch's code
5. **Create a pull request** when done
6. **Merge and move to the next branch**

### Step-by-Step Workflow

For each numbered branch:

```bash
# Switch to the branch
git checkout branch_number_i

# Get latest changes (except for branch_0)
git pull origin main
# Note: A VIM window might open - just type ":wq" to close it

# Push your branch
git push
```

Then:
1. 📖 Read the PR_i.md file carefully
2. 💻 Complete all the TODOs in the code
3. 🔧 Test your changes
4. 📤 Open **ONE** pull request to your main branch
5. ✅ Merge the pull request
6. 🔄 Move to the next branch

> **💡 Pro Tip**: Always integrate your previous work when starting a new branch (except branch_0)!

### 🔍 Understanding Pull Requests

Pull Requests (PRs) are how you propose and review changes before merging them into your main codebase. They're essential for team collaboration!

**Important**: When creating a PR, make sure you're merging into YOUR forked repository, not the original:

❌ **Wrong** (merging to original repo):
![PR Wrong](assets/PR_wrong.png)

✅ **Correct** (merging to your fork):
![PR Right](assets/PR_right.png)

## 💡 Development Tips

### Managing Dependencies

Use uv to manage dependencies. Install or update packages with:

```bash
uv add <package>==<version>
```

Then sync the environment and regenerate the dependency files:

```bash
uv sync
```

### MLflow Tracking

- **Issue**: with MLflow 3.x (locked at 3.17.2), the file-based tracking store (`./mlruns`, the default) is in maintenance mode and raises an error, so the usual `mlflow ui --backend-store-uri mlruns` command fails.
- **Resolution**: runs are stored in a local SQLite database, `mlflow.db` at the project root, and artifacts (models) go to the project-level `mlruns/` folder. The notebook sets both.
- **View the runs** from the project root:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- `mlflow.db`, `mlruns/` and `data/` are git-ignored and must not be pushed.

### Code Quality
- The pre-commit hooks will automatically format your code
- Remove all TODOs and unused code before final submission
- Use clear variable names and add docstrings

## 📊 Evaluation Criteria

Your project will be evaluated on:

### 🔍 **Code Quality**
- Clean, readable code structure
- Proper naming conventions
- Good use of docstrings and type hints

### 🎨 **Code Formatting**
- Consistent style (automated with pre-commit)
- Professional presentation

### ⚙️ **Functionality**
- Code runs without errors
- All requirements implemented correctly

### 📖 **Documentation & Reproducibility**
- Clear README with setup instructions
- Team member names and GitHub usernames
- Step-by-step instructions to run everything

### 🤝 **Collaboration**
- Effective use of Pull Requests
- Good teamwork and communication

---

## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [ ] Prefect workflows for model training
- [ ] Separate modules for training and inference
- [ ] Reproducible model and encoder generation

✅ **Automated Deployment**
- [ ] Prefect deployment for regular retraining

✅ **Production API**
- [ ] Working REST API for predictions
- [ ] Pydantic input validation
- [ ] Docker containerization

✅ **Professional Documentation**
- [ ] Updated README with team info
- [ ] Clear setup and run instructions
- [ ] All TODOs removed from code

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
