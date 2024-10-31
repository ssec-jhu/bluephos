
# Contributing to the BluePhos Pipeline Project

Thank you for your interest in contributing to the BluePhos pipeline project! As a contributor, you’ll work from your fork of the main repository. This document outlines the steps to set up your development environment, guidelines for coding, and instructions for submitting contributions.

## Repository Information

- **Main Repository**: [BluePhos Main Repo](https://github.com/ssec-jhu/bluephos.git)

## Getting Started

1. **Fork the Repository**: First, create a fork of the main repository in your own GitHub account. This will allow you to freely make changes without affecting the main repository.

2. **Clone Your Fork**: Clone your fork to your local machine:
   ```bash
   git clone https://github.com/your-username/bluephos.git
   cd bluephos
   ```

3. **Add the Main Repository as Upstream**: To keep your fork in sync with the latest updates from the main repo, add it as a second remote named `upstream`:
   ```bash
   git remote add upstream https://github.com/ssec-jhu/bluephos.git
   ```

4. **Create a Branch**: Create a new branch for each feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Making Changes

- **Coding Standards**: Follow PEP 8 for Python code. Use `ruff format` or `tox -e format` to ensure code formatting aligns with the repository’s requirements.
- **Testing**: Ensure your changes pass all tests. Include relevant tests for any new features you add.
- **Documentation**: Update the documentation for any significant code changes. This includes comments, docstrings, and relevant updates to the `README.md`.

## Keeping Your Fork Updated

Regularly pull updates from the main repository to keep your fork in sync:
```bash
git fetch upstream
git merge upstream/main
```

## Submitting a Pull Request

When you’re ready to contribute your changes:

1. **Commit Your Changes**: Write concise and descriptive commit messages.
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

2. **Push to Your Fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request**:
   - Go to your forked repository on GitHub.
   - Click on **New Pull Request**.
   - Ensure the base repository is `ssec-jhu/bluephos` and the base branch is `main`.
   - Provide a title and description for your pull request.
   - Submit the pull request for review.

## Code Review Process

Once you submit a pull request:
- The maintainers will review your changes and may request modifications.
- Please address any feedback and re-submit for review.

## Issues and Support

If you encounter issues or have questions, feel free to open an issue on GitHub. We’ll do our best to assist.

## License

By contributing, you agree that your contributions will be licensed under the same open-source license as the project.

---

Thank you for your contributions to the BluePhos pipeline! We’re excited to work with you.
