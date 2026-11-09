"""benchmark 데이터 받기의 네트워크 재시도 (`minipfn.benchmarks._retry`)."""

import openml.exceptions as oe
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


def _raise(error, *, cause=None, context=None, suppress=False):
    # `raise error from cause`, 또는 `context`를 처리하다 `error`가 난 것
    try:
        if context is not None:
            try:
                raise context
            except type(context):
                if suppress:
                    raise error from None
                raise error  # noqa: B904
        raise error from cause
    except type(error) as e:
        return e


def test_network_errors_are_recognised_through_the_cause_chain():
    assert is_network_error(requests.exceptions.ChunkedEncodingError("IncompleteRead"))
    protocol = urllib3.exceptions.ProtocolError("Connection broken")
    assert is_network_error(_raise(RuntimeError("wrapped"), cause=protocol))
    # 네트워크 오류를 처리하다 난 다른 오류는 그 오류 때문이 아니다
    assert not is_network_error(_raise(ValueError("bug"), context=TimeoutError()))
    assert not is_network_error(_raise(KeyError("x"), context=ConnectionError(), suppress=True))
    assert not is_network_error(ValueError("bad data"))
    # proxy·SSL 설정 오류는 기다려도 낫지 않는다
    assert not is_network_error(requests.exceptions.ProxyError("proxy"))
    assert not is_network_error(requests.exceptions.SSLError("cert"))


def test_openml_errors_are_retried_only_when_transient():
    def server_error(status):
        return oe.OpenMLServerError(
            f"Unexpected server error when calling https://x. Please contact the developers!\n"
            f"Status code: {status}\n<html>...</html>"
        )

    assert is_network_error(server_error(503))
    assert is_network_error(server_error(429))
    assert is_network_error(server_error(200))  # 본문이 깨짐
    assert not is_network_error(server_error(404))
    assert not is_network_error(server_error(403))
    assert not is_network_error(oe.OpenMLServerError("URI too long! (https://x)"))
    assert is_network_error(oe.OpenMLHashException("checksum"))
    # 없는 task 같은 API 응답(하위 클래스)은 다시 하지 않는다
    assert not is_network_error(oe.OpenMLServerException("Unknown task", code=151))

    class OpenMLServerError(Exception):  # 이름만 같은 다른 클래스
        pass

    assert not is_network_error(OpenMLServerError("Status code: 503"))


def test_retry_waits_and_tries_again(waits):
    errors = [requests.exceptions.ChunkedEncodingError("a"), ConnectionError("b")]
    fn, calls = _flaky(errors)
    assert retry(fn, what="t") == "ok"
    assert calls["n"] == 3 and waits == [30.0, 60.0]


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


def test_retry_attempts(waits):
    fn, calls = _flaky([TimeoutError()])
    with pytest.raises(TimeoutError):  # 한 번만 부른다
        retry(fn, what="t", attempts=1)
    assert calls["n"] == 1 and waits == []
    with pytest.raises(ValueError, match="attempts"):
        retry(fn, what="t", attempts=0)
    assert calls["n"] == 1


def test_load_openml_task_retries(monkeypatch, waits):
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

        def get_dataset(self, download_data):
            if not attempts:  # 첫 시도는 데이터를 받다가 끊긴다
                attempts.append(1)
                raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
            return Dataset()

    attempts = []
    monkeypatch.setattr(openml.tasks, "get_task", lambda task_id, download_splits: Task())
    task = amlb.load_openml_task(42)
    assert task.name == "toy" and task.task_type == "classification"
    assert waits == [30.0]


def test_load_openml_dataset_retries(monkeypatch, waits):
    import openml

    from minipfn.benchmarks import cc18

    class Dataset:
        name = "toy"
        default_target_attribute = "y"

        def get_data(self, target):
            if not attempts:  # 첫 시도는 parquet을 받다가 끊긴다
                attempts.append(1)
                raise urllib3.exceptions.ProtocolError("IncompleteRead")
            x = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["u", "v", "u"]})
            return x, pd.Series(["p", "q", "p"]), [False, True], ["a", "b"]

    attempts = []
    monkeypatch.setattr(openml.datasets, "get_dataset", lambda did, **kw: Dataset())
    ds = cc18.load_openml_dataset(3)
    assert ds.name == "toy" and ds.x.shape == (3, 2) and ds.categorical == [1]
    assert waits == [30.0]


def test_arena_prefetch_tasks_retries_each_task(monkeypatch, waits):
    from types import SimpleNamespace

    from minipfn.benchmarks import arena

    fetched = []
    materialized = []

    class Collection:
        # OpenML task 41(split 둘)·42와 Data Foundry task(id가 정수가 아니다)
        tasks = [SimpleNamespace(task_id_str=s) for s in ("41", "41", "42", "UserTask|abc")]

        def __iter__(self):
            return iter(self.tasks)

        def materialize(self):
            materialized.append(1)
            if len(materialized) == 1:
                raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
            return self

    jobs = ["job"]

    class TaskMetadata:
        def subset_to_jobs(self, js):
            assert js is jobs
            return Collection()

    context = SimpleNamespace(task_metadata_collection=TaskMetadata())
    monkeypatch.setattr(_retry, "fetch_openml_task", fetched.append)
    arena.prefetch_tasks(context, jobs, what="suite")
    assert fetched == [41, 42] and len(materialized) == 2 and waits == [30.0]
