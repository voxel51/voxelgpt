# Unit tests

## Running tests

```py
# All tests
pytest -q tests/

# All tests in a module
pytest -q tests/<filename>.py

# Single test
pytest -q tests/<filename.py> -k <test_name>
```

## Writing tests

- New test modules must start with `test_` or end with `_test.py`
- New test classes must start with `Test`
- New test functions must start with `test_`
