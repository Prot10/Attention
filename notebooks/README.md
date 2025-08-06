# Notebooks Directory

This directory hosts the Jupyter Notebooks for the **Data Science Project Template**. Jupyter Notebooks are essential for interactive development, data analysis, and documentation of insights.

## Structure

The `notebooks` directory is organized as follows:

- `exploratory/`: This sub-directory contains all exploratory notebooks where initial data analysis, data cleaning, and exploratory data analysis (EDA) are performed.
- `report/`: This sub-directory houses notebooks that are polished and contain insights, visualizations, and models ready for sharing with stakeholders.
- `archive/`: This sub-directory is for older notebooks or experiments that are kept for record-keeping but are not actively used.

## Usage Guidelines

1. **Interactive Exploration:** Use notebooks for interactive exploration and data visualization.
1. **Document Findings:** Document your findings, insights, and any assumptions in markdown cells to ensure that your analysis is understandable to others (and future-you).
1. **Version Control:** It's a good practice to version control your notebooks to keep track of the analysis evolution. However, remember to clear the output cells to reduce the file size before committing.
1. **Refactoring:** If some code in the notebook is to be used in production or repeated across different notebooks, it should be refactored into scripts within the `src/` directory of the project. Notebooks should primarily be used for exploration, and core code should reside in the `src/` directory to ensure modularity, reusability, and better version control.

______________________________________________________________________

The `notebooks` directory serves as a workspace for interactive analysis and communication of findings. Keeping it well-organized and version-controlled will streamline the analysis process and collaboration among team members.
