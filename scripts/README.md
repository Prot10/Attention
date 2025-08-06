# Scripts Directory

The `scripts` directory is designated for runnable scripts that execute the core code housed in the `src/` directory of the **Data Science Project Template**. These scripts are crucial for operationalizing the codebase, enabling reproducible runs for data preprocessing, experiments, and result generation.

The `scripts` directory can contain but is not limited to the following types of scripts:

- **Data Preprocessing:** Scripts to process raw data into a format suitable for analysis.
- **Experiments:** Scripts to run experiments, like model training and evaluation.
- **Result Generation:** Scripts to generate results, reports, or other output artifacts from the analysis.
- **Utilities:** Any other scripts that automate routine tasks or help manage the project.

## Usage Guidelines

1. **Documentation:** Document the scripts adequately, explaining the purpose, usage, expected inputs, and outputs.
1. **Version Control:** Version control scripts to track changes over time and ensure reproducibility.

## Running Scripts

To run a script, navigate to the root directory of the project, and use the following command:

```bash
uv run scripts/<script_name.py>
```

Replace `<script_name.py>` with the name of the script you want to run.

______________________________________________________________________

The `scripts` directory plays a pivotal role in operationalizing the codebase and ensuring the reproducibility and tractability of the project. Proper organization, documentation, and version control of scripts contribute to the project's overall maintainability and success.
