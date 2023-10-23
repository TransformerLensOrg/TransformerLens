def pytest_collectstart(collector):
    """Ignore several mimetypes when comparing notebooks."""
    if collector.fspath and collector.fspath.ext == ".ipynb":
        collector.skip_compare += (
            "text/html",
            "application/javascript",
            "application/vnd.plotly.v1+json",  # Plotly
        )
