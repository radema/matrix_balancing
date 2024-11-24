# For Contributors

<!-- Guidelines for contributors on code style, testing, and submitting pull requests. -->
Thank you for considering contributing to this project! To ensure that contributions are consistent, maintainable, and easy to integrate, please follow the guidelines below.

## Code Style

* **Python Version**: This project is developed with Python 3.7+ in mind. Ensure your code is compatible with Python 3.7 and later.
* **PEP 8**: Please follow PEP 8 for code style conventions. This includes proper indentation, line length (preferably 79 characters per line), and spacing around operators and parentheses.
* **Docstrings**: All functions, classes, and methods should include clear and concise docstrings that follow the Google Python Style Guide. Ensure that docstrings provide enough detail to explain the purpose and functionality of the code.
* **Type Hints**: Please include type hints where applicable. If you’re unsure about the correct types, refer to the Python typing documentation.

## Testing

* **Unit Tests**: Ensure that new code is covered by unit tests. The project uses pytest for testing, so make sure your tests are compatible with it.
* **Test Coverage**: Aim for high test coverage (preferably above 90%). New functionality should come with appropriate tests.
* **Running Tests**: You can run all the tests locally by executing `pytest` and ensuring that all tests pass before submitting a pull request.
* **Test Dependencies**: If you add new testing dependencies, ensure that they are included in the pyproject.toml file.

## Submitting Pull Requests

* **Fork the Repository**: Start by forking the repository and cloning it to your local machine.
* **Create a Feature Branch**: Create a new branch for your changes. Use a clear and descriptive name for the branch that reflects the nature of your work.
* **Make Changes**: Implement your changes and ensure that they are fully tested.
* **Commit Your Changes**: Write meaningful commit messages following this format:
```
    [TYPE] Short Description of the Change

    Detailed explanation of the changes if necessary.
```

    Replace `[TYPE]` with a type from the following options:

    * `feat`: A new feature
    * `fix`: A bug fix
    * `docs`: Documentation changes
    * `style`: Code style changes (formatting, missing semicolons, etc.)
    * `refactor`: Code refactoring (no new features, no bug fixes)
    * `test`: Adding or updating tests
    * `chore`: Maintenance tasks (build scripts, dependencies)

* **Push Changes**: Push your changes to your forked repository.
* **Open a Pull Request**: Once your changes are pushed, open a pull request (PR) to the main repository. Be sure to describe your changes and why they are needed in the PR description.
* **Review and Feedback**: The maintainers will review your pull request. Be prepared to make changes if necessary based on feedback.
* **Merge**: After approval, your changes will be merged into the main repository. Ensure your branch is up to date with the base branch before merging.

## Additional Notes

**Issues**: Before starting a new feature or bug fix, it’s a good idea to open an issue to discuss the change with the maintainers. This helps avoid duplication of effort and ensures alignment with the project’s goals.
**Changelog**: If your contribution introduces significant changes, be sure to update the CHANGELOG.md file to reflect those changes, including versioning information.

By following these guidelines, you help maintain a clean and efficient development process for this project. We look forward to your contributions!