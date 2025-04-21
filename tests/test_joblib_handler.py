"""Test the custom joblib handler."""

import zoneinfo
from datetime import datetime

import joblib
import pytest
import sklearn
import time_machine
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from lazyscribe_joblib import JoblibArtifact


@time_machine.travel(
    datetime(2025, 1, 20, 13, 23, 30, tzinfo=zoneinfo.ZoneInfo("UTC")), tick=False
)
def test_joblib_handler(tmp_path):
    """Test reading and writing scikit-learn estimators with the joblib handler."""
    # Fit a basic estimator
    X, y = make_classification(n_samples=100, n_features=10)
    estimator = SVC(kernel="linear")
    estimator.fit(X, y)

    # Construct the handler and write the estimator
    location = tmp_path / "my-estimator-location"
    location.mkdir()
    handler = JoblibArtifact.construct(name="My estimator", value=estimator)

    assert (
        handler.fname
        == f"my-estimator-{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
    )

    with open(location / handler.fname, "wb") as buf:
        handler.write(estimator, buf)

    assert (location / handler.fname).is_file()

    # Read the estimator back and ensure it's fitted
    with open(location / handler.fname, "rb") as buf:
        out = handler.read(buf)

    sklearn.utils.validation.check_is_fitted(out)

    # Check that the handler correctly captures the environment variables
    assert (
        JoblibArtifact(
            name="EXCLUDED FROM COMPARISON",
            fname="EXCLUDED FROM COMPARISON",
            value=None,
            created_at=None,
            writer_kwargs=None,
            package="sklearn",
            package_version=sklearn.__version__,
            joblib_version=joblib.__version__,
            version=None,
            dirty=False,
        )
    ) == handler


def test_joblib_handler_error_no_inputs():
    """Test that the joblib handler raises an error when no value or package is provided."""
    with pytest.raises(ValueError):
        _ = JoblibArtifact.construct(name="My artifact")


def test_joblib_handler_invalid_package():
    """Test that the joblib handler raises an error when an invalid package is provided."""
    with pytest.raises(ValueError):
        _ = JoblibArtifact.construct(name="My artifact", package="my_invalid_package")


def test_joblib_handler_raise_attribute_error():
    """Test that the joblib handler raises an error for objects where the package can't be determined."""
    numpy = pytest.importorskip("numpy")

    myarr = numpy.array([])
    with pytest.raises(AttributeError):
        JoblibArtifact.construct(name="My array", value=myarr)
