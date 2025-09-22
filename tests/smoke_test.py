"""Check that basic features work.

Used in our publishing pipeline."""

import tempfile
from pathlib import Path

from lazyscribe import Project
from lazyscribe.linked import LinkedList

with tempfile.TemporaryDirectory() as tmpdir:
    # Create a project
    project = Project(Path(tmpdir) / "project.json", mode="w")
    with project.log(name="Joblib experiment") as exp:
        # Create a fake linked list so we have an object that has an associated module
        obj = LinkedList()
        obj.append(["a", "b", "c"])
        exp.log_artifact(name="feature-names", value=obj, handler="joblib")

    project.save()

    exp = project["joblib-experiment"]
    value = exp.load_artifact(name="feature-names")

    assert value.head.data == ["a", "b", "c"]
