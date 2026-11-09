"""benchmark 데이터를 받다가 네트워크가 끊겨도 평가 전체가 멈추지 않도록 다시 시도한다.

AI Runtime 평가에서 OpenML 데이터를 받는 도중 연결이 끊겨(`IncompleteRead`) 수십 분 돈 평가가
처음부터 다시 돌아야 했던 적이 여러 번 있다(#29). 네트워크 오류일 때만 잠깐 기다렸다가 다시 하고,
데이터가 없다는 응답 같은 다른 오류는 그대로 낸다.
"""

from __future__ import annotations

import http.client
import shutil
import time
from collections.abc import Callable

# 다시 해 볼 횟수와 첫 대기 시간(초). 대기는 매번 두 배 (30, 60, 120초)
ATTEMPTS = 4
FIRST_WAIT = 30.0


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


def is_network_error(error: BaseException) -> bool:
    """네트워크가 끊기거나 서버가 잠깐 실패한 오류인지 (원인 사슬까지 본다).

    OpenML의 `OpenMLServerError`(예기치 않은 서버 오류)와 `OpenMLHashException`(받은 파일이
    깨짐)은 다시 할 만하다. 그 하위 클래스인 `OpenMLServerException`은 "없는 task" 같은 API
    응답이라 다시 하지 않는다."""
    types = _network_error_types()
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, types):
            return True
        if type(error).__name__ in ("OpenMLServerError", "OpenMLHashException"):
            return True
        error = error.__cause__ or error.__context__
    return False


def retry[T](
    fn: Callable[[], T],
    *,
    what: str,
    attempts: int = ATTEMPTS,
    wait: float = FIRST_WAIT,
    before_retry: Callable[[], None] | None = None,
) -> T:
    """`fn()`을 부르고, 네트워크 오류(`is_network_error`)면 기다렸다가 `attempts`번까지 다시
    한다. `before_retry`는 다시 하기 전에 부른다(받다 만 캐시 지우기 등). 다른 오류이거나 마지막
    시도도 실패하면 그 오류를 그대로 낸다."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts or not is_network_error(e):
                raise
            print(
                f"{what}: 네트워크 오류로 {wait:.0f}초 뒤 다시 ({attempt}/{attempts - 1}):"
                f" {type(e).__name__}: {e}",
                flush=True,
            )
            if before_retry is not None:
                before_retry()
            time.sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")


def clear_openml_cache(kind: str, id_: int | None) -> None:
    """OpenML 캐시에서 그 dataset·task 폴더를 지운다(받다 만 파일을 다음 시도가 읽지 않도록).
    `kind`는 "datasets" 또는 "tasks"."""
    if id_ is None:
        return
    from openml.utils import _get_cache_dir_for_id

    shutil.rmtree(_get_cache_dir_for_id(kind, int(id_)), ignore_errors=True)
