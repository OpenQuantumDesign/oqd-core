# How to contribute to Open Quantum Design

Thank you for considering contributing to OQD! We welcome contributions, including raising issues, bug fixes, and feature additions. Here you will find details on how to install the repo locally for development and how to 


## Support questions

Please don't use the issue tracker for this. The issue tracker is a tool
to address bugs and feature requests in OQD's repositories. 

<!-- Use one of the following resources for questions: -->
<!-- -   The `#oqd` channel on the Unitary Fund Discord: -->


## Reporting issues

Please follow the [`ISSUE_TEMPLATE.md`](https://github.com/OpenQuantumDesign/oqd-core/blob/main/.github/ISSUE_TEMPLATE.md)
Include the following information in your post:

-   Describe what you expected to happen.
-   If possible, include a minimal reproducible example to help us
    identify the issue. This also helps check that the issue is not with
    your own code.
-   Describe what actually happened. Include the full traceback if there
    was an exception.
-   List your version, Python version, and OS. 


## Submitting feature

If there is not an open issue for what you want to submit, prefer
opening one for discussion before working on a PR. You can work on any
issue that doesn't have an open PR linked to it or a maintainer assigned
to it. These show up in the sidebar. No need to ask if you can work on
an issue that interests you. Please follow the [`PULL_REQUEST_TEMPLATE.md`](https://github.com/OpenQuantumDesign/oqd-core/blob/main/.github/PULL_REQUEST_TEMPLATE.md)

Include the following in your patch:

-   Use `ruff` to format your code, using `ruff format`
-   Include tests if your PR adds or changes code.
-   Update any relevant documentation pages and docstrings.


## First time setup

- Download and install the latest version of [`git`]( https://git-scm.com/downloads).
- [Fork](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) the repository to your own account
- Clone the main repository locally.
    ```bash
    git clone https://github.com/{username}/oqd-core
    cd oqd-core
    ```
    where `{username}` is your Github username.
- Create a virtual environment.
    ```bash
    python3 -m venv .venv
    . env/bin/activate
    ```
    On Windows, activating is different.

    ```
    env\Scripts\activate
    ```

-   Install the development dependencies
    ```bash
    pip install -e .[docs,tests]
    ```


## Start coding

- Create a branch to work on
    ```bash
    git branch BRANCH-NAME
    git checkout BRANCH-NAME
    ```
-   Using your favorite editor, make your changes, committing as you go.
-   Include tests that cover any code changes you make. Make sure the
    test fails without your patch. Run the tests as described below.
-   Push your commits to your fork on GitHub and
    [create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). Link to the issue being addressed with
    ``fixes #123`` in the pull request.

## Running the tests

Run the basic test suite with pytest.

```
pytest
```
This runs the tests for the current environment, which is usually
sufficient. CI will run the full suite when you submit your pull
request. 

## Running test coverage

Generating a report of lines that do not have test coverage can indicate
where to start contributing. Run ``pytest`` using [`coverage`](https://coverage.readthedocs.io) and
generate a report.

```bash
pip install coverage
coverage run -m pytest
coverage html
```
Open `htmlcov/index.html` in your browser to explore the report.



## Building the docs
Ensure the documentation dependencies are installed with
```bash
pip install .[docs] 
```

Build the docs in the `docs` directory using [Mkdocs](https://mkdocs.org).
```bash
cd docs
mkdocs serve
```
The documentation should automatically be opened in the browser.
