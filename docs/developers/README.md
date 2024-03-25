# synthx for developers

## Pre-requisites

- Poetry
- Python >= 3.9
- GCC Make

## Set up

Clone the repository to local

```
git clone git@github.com:kenki931128/synthx.git
```

```
cd synthx
poetry install
```

## How to run

You can start up the jupyter notebook with the command below, and try the library

```
make notebook
```

## How to contribute




# synthx for developers

## Pre-requisites

- Poetry
- Python >= 3.9
- GCC Make

## Set up
Clone the repository to local

```
git clone git@github.com:kenki931128/synthx.git
```

Install the required dependencies using Poetry

```
cd synthx
poetry install
```

## How to run
You can start up the Jupyter Notebook with the command below and try out the library

```
make notebook
```

## How to contribute

Before submitting a pull request or pushing the commits, make sure to run the following commands to check your code:

```
make lint
make test
```

These commands will run linting and testing on your code to ensure it follows the project's coding standards and passes all tests.

To contribute to the project, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch from the develop branch for your changes.
3. Make your changes with proper tests and commit them with clear and descriptive commit messages.
4. Add comments to your code to improve readability and understanding.
5. Push your changes to your forked repository.
6. Open a pull request from your branch to the develop branch of the main repository.

Please provide a detailed description of your changes in the pull request, explaining the purpose and any relevant information.

**Code Review Process**

Once you have submitted a pull request, the project maintainers will review your code. They may provide feedback or request changes. Please be responsive to their comments and make the necessary adjustments.

After your code has been reviewed and approved, it will be merged into the develop branch.

**Release Process**

The project maintainers will periodically merge the develop branch into the main branch for releases. Before a release, the following steps will be taken:

1. All tests will be run to ensure the code is functioning as expected.
2. The documentation will be updated to reflect any changes or new features.
3. A new version number will be assigned following the Semantic Versioning (SemVer) guidelines.
4. A new release will be created on GitHub with release notes detailing the changes.

**Reporting Issues**

If you encounter any bugs, issues, or have suggestions for improvements, please open an issue on the GitHub repository. Provide as much detail as possible, including steps to reproduce the issue, expected behavior, and actual behavior.
