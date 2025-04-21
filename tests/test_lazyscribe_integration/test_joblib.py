"""Test using the joblib handler with Lazyscribe."""

import zoneinfo
from datetime import datetime

import sklearn
import time_machine
from lazyscribe import Project
from sklearn.datasets import make_classification
from sklearn.svm import SVC


@time_machine.travel(
    datetime(2025, 1, 20, 13, 23, 30, tzinfo=zoneinfo.ZoneInfo("UTC")), tick=False
)
def test_joblib_project_write(tmp_path):
    """Test logging an artifact using the Joblib handler."""
    location = tmp_path / "my-project-location"
    location.mkdir()

    project = Project(fpath=location / "project.json", mode="w")
    with project.log("My experiment") as exp:
        X, y = make_classification(n_samples=100, n_features=10)
        estimator = SVC(kernel="linear")
        estimator.fit(X, y)

        exp.log_artifact(name="My estimator", value=estimator, handler="joblib")

    project.save()

    assert (
        location / "my-experiment-20250120132330" / "my-estimator-20250120132330.joblib"
    ).is_file()

    project_r = Project(fpath=location / "project.json", mode="r")
    out = project_r["my-experiment"].load_artifact(name="My estimator")

    sklearn.utils.validation.check_is_fitted(out)
