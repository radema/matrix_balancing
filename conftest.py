import logging
import os
from datetime import datetime

import pytest

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


def pytest_sessionstart(session):
    """
    Hook to run at the start of the pytest session.
    Configures initial logging setup.
    """
    logging.info("========== TEST SESSION START ==========")
    logging.info(f"Test session started at {datetime.now()}")


def pytest_sessionfinish(session, exitstatus):
    """
    Hook to run at the end of the pytest session.
    Logs summary information.
    """
    logging.info("========== TEST SUMMARY ==========")
    logging.info(f"Exit status: {exitstatus}")
    if exitstatus != pytest.ExitCode.OK:
        logging.error("Some tests failed or encountered errors.")
    else:
        logging.info("All tests passed successfully!")
    logging.info(f"Test session ended at {datetime.now()}")
    logging.info(f"Test report saved to {LOG_FILE}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Hook to add a summary report to the log file after tests finish.
    """
    summary = terminalreporter.stats
    total_tests = sum(len(items) for items in summary.values())
    logging.info(f"Total tests run: {total_tests}")
    logging.info(f"Failures: {len(summary.get('failed', []))}")
    logging.info(f"Errors: {len(summary.get('error', []))}")
    logging.info(f"Passed: {len(summary.get('passed', []))}")

    if "failed" in summary:
        logging.error("Failures:")
        for report in summary["failed"]:
            logging.error(f"Test: {report.nodeid}")
            logging.error(f"Error:\n{report.longrepr}")
