import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from app import worker


class _JobQuery:
    def __init__(self, job):
        self.job = job

    def filter(self, *_args, **_kwargs):
        return self

    def first(self):
        return self.job


class _Session:
    def __init__(self, job):
        self.job = job
        self.closed = False

    def query(self, _model):
        return _JobQuery(self.job)

    def close(self):
        self.closed = True


class WorkerQueueTests(unittest.TestCase):
    def test_terminal_job_is_not_executed_again(self):
        job = SimpleNamespace(status="succeeded")
        session = _Session(job)

        with (
            patch.object(worker, "SessionLocal", return_value=session),
            patch.object(worker, "_set") as set_job,
        ):
            worker._run_job(uuid.uuid4())

        set_job.assert_not_called()
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
