"""benchmark 데이터 받기의 네트워크 재시도 (`minipfn.benchmarks._retry`)."""

import pandas as pd
import pytest
import requests
import urllib3

from minipfn.benchmarks import _retry
from minipfn.benchmarks._retry import is_network_error, retry


@pytest.fixture
def waits(monkeypatch):
    out = []
    monkeypatch.setattr(_retry.time, "sleep", out.append)
    return out


def _flaky(errors, value="ok"):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return value

    return fn, calls


def test_network_errors_are_recognised_through_the_cause_chain():
    import openml.exceptions as oe

    assert is_network_error(requests.exceptions.ChunkedEncodingError("IncompleteRead"))
    try:
        try:
            raise urllib3.exceptions.ProtocolError("Connection broken")
        except urllib3.exceptions.ProtocolError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert is_network_error(outer)
    assert is_network_error(oe.OpenMLServerError("Unexpected server error"))
    # 없는 task 같은 API 응답(하위 클래스)과 다른 오류는 다시 하지 않는다
    assert not is_network_error(oe.OpenMLServerException("Unknown task", code=151))
    assert not is_network_error(ValueError("bad data"))


def test_retry_waits_and_tries_again(waits):
    errors = [requests.exceptions.ChunkedEncodingError("a"), ConnectionError("b")]
    cleared = []
    fn, calls = _flaky(errors)
    assert retry(fn, what="t", before_retry=lambda: cleared.append(1)) == "ok"
    assert calls["n"] == 3 and waits == [30.0, 60.0] and len(cleared) == 2


def test_retry_raises_other_errors_at_once(waits):
    fn, calls = _flaky([ValueError("bad")])
    with pytest.raises(ValueError):
        retry(fn, what="t")
    assert calls["n"] == 1 and waits == []


def test_retry_gives_up_after_the_last_attempt(waits):
    fn, calls = _flaky([TimeoutError(str(i)) for i in range(5)])
    with pytest.raises(TimeoutError):
        retry(fn, what="t", attempts=3)
    assert calls["n"] == 3 and waits == [30.0, 60.0]


def test_load_openml_task_retries_and_clears_the_partial_cache(monkeypatch, waits):
    import openml

    from minipfn.benchmarks import amlb

    class Dataset:
        name = "toy"

        def get_data(self, target, dataset_format):
            return pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([0, 1]), [False], ["a"]

    class Task:
        dataset_id = 7
        target_name = "y"
        task_type_id = openml.tasks.TaskType.SUPERVISED_CLASSIFICATION

        def get_dataset(self):
            if not attempts:  # 첫 시도는 데이터를 받다가 끊긴다
                attempts.append(1)
                raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
            return Dataset()

    attempts, cleared = [], []
    monkeypatch.setattr(openml.tasks, "get_task", lambda task_id, download_splits: Task())
    monkeypatch.setattr(_retry, "clear_openml_cache", lambda kind, id_: cleared.append((kind, id_)))
    task = amlb.load_openml_task(42)
    assert task.name == "toy" and task.task_type == "classification"
    assert cleared == [("tasks", 42), ("datasets", 7)] and waits == [30.0]
