# Running Unit Tests
To Run Single Test: pytest -q tests/<filename.py> -k <test_name>
To Run All Tests From Single Test File: pytest -q tests/<filename>.py
To Run All Unit Tests: pytest -q tests/

# Rules
To add a new test, the name must start with test_
To add a new test class, the name must start with Test
To add a new test file, the name must start with test_ or end with _test