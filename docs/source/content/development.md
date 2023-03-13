## Local Development

### DevContainer

For a one-click setup of your development environment, this project includes a [DevContainer](https://containers.dev/). It can be used locally with [VS Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) or with [GitHub Codespaces](https://github.com/features/codespaces).

### Manual Setup

This project uses [Poetry](https://python-poetry.org/docs/#installation) for package management. Install as follows (this will also setup your virtual environment):

```bash
poetry config virtualenvs.in-project true
poetry install --with dev
```

Optionally, if you want Jupyter Lab you can run `poetry run pip install jupyterlab` (to install in the same virtual environment), and then run with `poetry run jupyter lab`.

Then the library can be imported as `import transformer_lens`.

### Testing

If adding a feature, please add unit tests for it to the tests folder, and check that it hasn't broken anything major using the existing tests (install pytest and run it in the root TransformerLens/ directory).

To run tests, you can use the following command:

```
poetry run pytest -v transformer_lens/tests
```
