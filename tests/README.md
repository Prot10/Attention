# Tests Directory

This directory contains all the tests for the **Data Science Project Template**. Staying true to best practices, we aim for at least 80% test coverage to ensure the accuracy and reliability of the analysis and models.

## Structure

The structure of the `tests` directory is as follows:

- `unit`: This directory contains unit tests which test functions, methods, and classes in isolation.
- `integration`: This directory contains integration tests which test the interaction between different parts of the project.

Each directory should contain an `__init__.py` file to ensure the directory is treated as a Python package.

## conftest.py

The `conftest.py` file in this directory is used to provide configuration settings for pytest, and to define fixtures that can be reused across multiple test files. This is where common setup code and variables can be placed.

## How to Run Tests

To run all tests, navigate to the root directory of the project and run the following command:

```bash
pytest
```

You can also run tests in a specific directory by specifying the path, like so:

```bash
pytest tests/unit
```

## Adding Tests

When adding tests, it's important to follow a naming convention. A common convention is to prefix test file names with `test_`, for example `test_data_preprocessing.py`. Within test files, name individual test functions with a `test_` prefix as well, like `def test_remove_outliers()`.

It's good practice to have a descriptive docstring in each test function explaining what aspect of the function/method/class is being tested.

## Mocking with pytest

`pytest` offers a powerful mocking facility called `pytest.mock` which is used to replace parts of your system under test and assert that they were used in the expected way. This is particularly useful when you have external dependencies or parts of your system that are not ready or are expensive in terms of time/resources.

Example of using `pytest.mock`:

```python
def test_some_function(mocker):
    mocker.patch("some_module.some_function", return_value=3)
    assert some_module.some_function(4) == 3
```

______________________________________________________________________

The `tests` directory is crucial for ensuring the correctness of the data science project. As the project evolves, ensure that the tests are updated to reflect the changes in the project codebase.
