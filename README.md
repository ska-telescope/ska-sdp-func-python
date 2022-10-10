# SKA SDP Python-based Processing Functions

This repository contains processing functions implemented in Python.
These functions were originally developed in [RASCIL](https://gitlab.com/ska-telescope/external/rascil).
They were migrated during PI16 (autumn 2022) to provide an independent library 
and easy access to the processing scripts.

## Standard CI machinery

This repository is set up to use the
[Makefiles](https://gitlab.com/ska-telescope/sdi/ska-cicd-makefile) and [CI
jobs](https://gitlab.com/ska-telescope/templates-repository) maintained by the
System Team. For any questions, please look at the documentation in those
repositories or ask for support on Slack in the #team-system-support channel.

To keep the Makefiles up to date in this repository, follow the instructions
at: https://gitlab.com/ska-telescope/sdi/ska-cicd-makefile#keeping-up-to-date

## Contributing to this repository

[Black](https://github.com/psf/black), [isort](https://pycqa.github.io/isort/),
and various linting tools are used to keep the Python code in good shape.
Please check that your code follows the formatting rules before committing it
to the repository. You can apply Black and isort to the code with:

```bash
make python-format
```

and you can run the linting checks locally using:

```bash
make python-lint
```

The linting job in the CI pipeline does the same checks, and it will fail if
the code does not pass all of them.

## Creating a new release

When you are ready to make a new release (maintainers only):

  - Check out the master branch
  - Create an issue in the [Release Management](https://jira.skatelescope.org/projects/REL/summary) project
  - Update the version number in `.release` with
    - `make bump-patch-release`,
    - `make bump-minor-release`, or
    - `make bump-major-release`
  - Set the Python package version number with `make python-set-release`
  - Create the git tag with `make git-create-tag`
  - Push the changes with `make git-push-tag`
