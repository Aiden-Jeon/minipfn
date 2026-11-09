"""benchmark 데이터를 받다가 네트워크가 끊겨도 평가 전체가 멈추지 않도록 다시 시도한다.

AI Runtime 평가에서 OpenML 데이터를 받는 도중 연결이 끊겨(`IncompleteRead`) 수십 분 돈 평가가
처음부터 다시 돌아야 했던 적이 여러 번 있다(#29). 네트워크 오류일 때만 잠깐 기다렸다가 다시 하고,
데이터가 없다는 응답 같은 다른 오류는 그대로 낸다.

openml도 API 요청 하나를 몇 번(기본 5번, 1~4초 간격) 다시 하지만 그보다 긴 끊김은 그대로
올라오고, parquet 내려받기(minio)는 다시 하지 않는다. 그래서 그 바깥에서 더 길게 기다린다.

받다 만 캐시는 지우지 않는다. openml은 `get_task`·`get_dataset`이 실패하면 그 캐시 폴더를 스스로
지우고, API 응답은 다 받은 뒤에 파일로 쓰며, parquet은 `*.part.minio`에 받은 뒤 이름을 바꾼다
(다음 시도는 거기서 이어 받는다).
"""

from __future__ import annotations

import http.client
import re
import time
from collections.abc import Callable

# 시도 횟수와 첫 대기 시간(초). 대기는 매번 두 배 (30, 60, 120초)
ATTEMPTS = 4
FIRST_WAIT = 30.0

# openml `OpenMLServerError` 메시지의 HTTP 상태
_STATUS = re.compile(r"Status code: (\d{3})")


def _network_error_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [ConnectionError, TimeoutError, http.client.IncompleteRead]
    try:
        import requests

        types += [
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.Timeout,
        ]
    except ImportError:
        pass
    try:
        import urllib3

        types += [urllib3.exceptions.ProtocolError, urllib3.exceptions.TimeoutError]
    except ImportError:
        pass
    return tuple(types)


def _config_error_types() -> tuple[type[BaseException], ...]:
    # requests의 `ConnectionError` 가운데 proxy·SSL 오류는 설정 문제라 기다려도 낫지 않는다
    try:
        import requests
    except ImportError:
        return ()
    return (requests.exceptions.ProxyError, requests.exceptions.SSLError)


def _is_transient_openml_error(error: BaseException) -> bool:
    try:
        import openml.exceptions as oe
    except ImportError:
        return False
    if isinstance(error, oe.OpenMLHashException):  # 받은 파일이 깨짐
        return True
    # 하위 클래스(`OpenMLServerException`: "없는 task" 같은 API 응답)는 다시 하지 않는다
    if type(error) is not oe.OpenMLServerError:
        return False
    # 상태가 200(본문이 깨짐)·429·5xx일 때만. 4xx(없거나 권한이 없는 데이터)와 상태가 없는
    # 것("URI too long")은 다시 해도 같다
    m = _STATUS.search(str(error))
    return m is not None and (int(m.group(1)) in (200, 429) or int(m.group(1)) >= 500)


def is_network_error(error: BaseException) -> bool:
    """네트워크가 끊기거나 서버가 잠깐 실패한 오류인지.

    명시적 원인(`raise ... from e`)의 사슬을 따라간다. 다른 오류를 처리하다 난 오류
    (`__context__`)는 그 오류 때문이라고 보지 않는다. OpenML 오류는 `OpenMLHashException`(받은
    파일이 깨짐)과 상태가 200·429·5xx인 `OpenMLServerError`(예기치 않은 서버 오류)만 다시 한다."""
    network, config = _network_error_types(), _config_error_types()
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, config):
            return False
        if isinstance(error, network) or _is_transient_openml_error(error):
            return True
        error = error.__cause__
    return False


def retry[T](
    fn: Callable[[], T], *, what: str, attempts: int = ATTEMPTS, wait: float = FIRST_WAIT
) -> T:
    """`fn()`을 부르고, 네트워크 오류(`is_network_error`)면 기다렸다가 모두 `attempts`번까지
    부른다. 다른 오류이거나 마지막 시도도 실패하면 그 오류를 그대로 낸다."""
    if attempts < 1:
        raise ValueError(f"attempts는 1 이상이어야 한다: {attempts}")
    for attempt in range(1, attempts):
        try:
            return fn()
        except Exception as e:
            if not is_network_error(e):
                raise
            print(
                f"{what}: 네트워크 오류로 {wait:.0f}초 뒤 다시 ({attempt}/{attempts - 1}):"
                f" {type(e).__name__}: {e}",
                flush=True,
            )
            time.sleep(wait)
            wait *= 2
    return fn()


def fetch_openml_task(task_id: int):
    """OpenML task와 split, dataset(설명·feature·데이터)을 받는다(캐시에 남는다). 네트워크
    오류면 다시 한다. `(task, dataset, dataset.get_data(target)의 결과)`를 돌려준다."""
    import openml

    def fetch():
        task = openml.tasks.get_task(task_id, download_splits=True)
        dataset = task.get_dataset(download_data=True)
        return task, dataset, dataset.get_data(target=task.target_name, dataset_format="dataframe")

    return retry(fetch, what=f"OpenML task {task_id}")
