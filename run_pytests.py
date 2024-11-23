import os
import pytest
import logging
import sys
from datetime import datetime

# Add 'src' directory to the Python path
SRC_DIR = os.path.abspath("src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Define paths
LOG_DIR = "logs/test_logs"
LOG_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"test_report_{LOG_TIMESTAMP}.log")
TESTS_DIR = "tests"

# Ensure the logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    # Run pytest and capture the result
    PYTEST_PARAM = ["--maxfail=5", "--disable-warnings", "--tb=short"]
    result = pytest.main([TESTS_DIR] + PYTEST_PARAM)

    # Write a summary to the log file
    logging.info("========== TEST SUMMARY ==========")
    logging.info(f"Tests run: {result.testscollected}")
    logging.info(f"Errors: {len(result.errors)}")
    logging.info(f"Failures: {len(result.failures)}")

    if result.exitstatus != pytest.ExitCode.OK:
        logging.error("Some tests failed or encountered errors.")
        if result.failures:
            logging.error("Failures:")
            for test, err in result.failures:
                logging.error(f"Test: {test}\nError:\n{err}")
        if result.errors:
            logging.error("Errors:")
            for test, err in result.errors:
                logging.error(f"Test: {test}\nError:\n{err}")
    else:
        logging.info("All tests passed successfully!")


if __name__ == "__main__":
    main()
    print(f"Test report saved to {LOG_FILE}")
