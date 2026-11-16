"""bagging의 bag을 GPU마다 하나씩 띄운 worker 프로세스에 나눠 fit·예측한다 (평가 스크립트의
`--bag-gpus`).

`--shards`는 데이터셋을 GPU에 나눠 주므로 가장 큰 데이터셋 하나가 GPU 한 장에서 bag k개를
차례로 돌고, 그동안 다른 GPU는 논다. 여기서는 split 하나의 bag k개를 worker들에 동시에 보내고
예측도 동시에 받는다. 데이터셋은 차례로 돌지만 split마다 GPU를 모두 쓴다.

- worker는 `spawn`으로 띄운 프로세스이고 `CUDA_VISIBLE_DEVICES`로 GPU 하나만 본다. worker는
  `script`(파일 경로)를 불러와 `builder(argv)`로 estimator를 만드는 함수 `make(problem_type,
  categorical)`를 한 번 만들고(모델을 한 번만 읽는다), 맞춘 estimator를 번호로 들고 있다
- bag i는 worker i % n에 간다. bag 수가 worker 수보다 많으면 worker 수만큼씩 나눠 보낸다
- worker 안에서 fit·예측이 예외를 내면 그 split만 실패한다(같은 프로세스에서 돌릴 때와 같다).
  worker 프로세스가 죽으면 평가를 멈춘다(`SystemExit`: 나머지 split이 조용히 imputed가 되지 않게)
"""

from __future__ import annotations

import importlib.util
import itertools
import multiprocessing as mp
import os
import traceback
from collections.abc import Sequence
from typing import Any

import numpy as np


def _worker(script: str, builder: str, argv: list[str], conn) -> None:
    spec = importlib.util.spec_from_file_location("_minipfn_bag_worker_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    make = getattr(module, builder)(argv)
    fitted: dict[int, Any] = {}
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        op = msg[0]
        if op == "stop":
            return
        if op == "drop":
            for key in msg[1]:
                fitted.pop(key, None)
            continue
        try:
            if op == "fit":
                _, key, problem_type, categorical, X, y, groups = msg
                kwargs = {} if groups is None else {"groups": groups}
                estimator = make(problem_type, categorical).fit(X, y, **kwargs)
                fitted[key] = estimator
                classes = getattr(estimator, "classes_", None)
                conn.send(("ok", None if classes is None else np.asarray(classes)))
            elif op == "predict":
                _, key, X, proba = msg
                estimator = fitted[key]
                out = estimator.predict_proba(X) if proba else estimator.predict(X)
                conn.send(("ok", np.asarray(out)))
            else:
                conn.send(("error", f"모르는 요청: {op}"))
        except Exception:  # noqa: BLE001 (그 split만 실패로 돌려준다)
            conn.send(("error", traceback.format_exc()))


class BagPool:
    """GPU(`devices`, `CUDA_VISIBLE_DEVICES` 값)마다 worker 하나. `argv`는 worker마다 builder에 넘길
    인자이고(`argv[i]`), `threads`는 worker의 CPU thread 수다."""

    def __init__(
        self,
        script: str,
        builder: str,
        argv: Sequence[list[str]],
        devices: Sequence[str],
        threads: int = 1,
    ) -> None:
        if len(argv) != len(devices) or not devices:
            raise ValueError("worker마다 argv와 GPU를 하나씩 준다")
        ctx = mp.get_context("spawn")
        self.workers = []
        self._keys = itertools.count()
        for dev, args in zip(devices, argv, strict=True):
            parent, child = ctx.Pipe()
            # spawn한 프로세스는 띄울 때의 환경 변수를 받는다
            saved = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
            os.environ.update(CUDA_VISIBLE_DEVICES=str(dev), OMP_NUM_THREADS=str(threads))
            try:
                process = ctx.Process(
                    target=_worker, args=(script, builder, list(args), child), daemon=True
                )
                process.start()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            child.close()
            self.workers.append((process, parent))

    def __len__(self) -> int:
        return len(self.workers)

    def _recv(self, i: int):
        process, conn = self.workers[i]
        try:
            status, value = conn.recv()
        except (EOFError, OSError) as e:
            raise SystemExit(
                f"bag worker {i}가 죽었다 (exit code {process.exitcode}): {e!r}"
            ) from e
        if status != "ok":
            raise RuntimeError(f"bag worker {i}:\n{value}")
        return value

    def _send(self, i: int, msg) -> None:
        process, conn = self.workers[i]
        try:
            conn.send(msg)
        except (BrokenPipeError, OSError) as e:
            raise SystemExit(
                f"bag worker {i}가 죽었다 (exit code {process.exitcode}): {e!r}"
            ) from e

    def _rounds(self, requests: list) -> list:
        """요청 i를 worker i % n에 worker 수만큼씩 보내고 답을 요청 순서대로 모은다. 한 round의
        답을 다 받은 뒤에 실패를 알린다(worker에 받지 않은 답이 남지 않게)."""
        n, out = len(self.workers), []
        for start in range(0, len(requests), n):
            chunk = requests[start : start + n]
            for j, msg in enumerate(chunk):
                self._send((start + j) % n, msg)
            answers, error = [], None
            for j in range(len(chunk)):
                try:
                    answers.append(self._recv((start + j) % n))
                except RuntimeError as e:
                    error = error or e
            if error is not None:
                raise error
            out.extend(answers)
        return out

    def fit(
        self, problem_type: str, categorical: list[int], parts: list[tuple[Any, Any, Any]]
    ) -> tuple[list[tuple[int, int]], list]:
        """bag마다 (X, y, groups)를 맞춘다. (bag마다 (worker, 번호), bag마다 `classes_` 또는
        None)."""
        n = len(self.workers)
        keys = [next(self._keys) for _ in parts]
        handles = [(i % n, key) for i, key in enumerate(keys)]
        requests = [
            ("fit", key, problem_type, categorical, X, y, groups)
            for key, (X, y, groups) in zip(keys, parts, strict=True)
        ]
        try:
            classes = self._rounds(requests)
        except RuntimeError:
            self.drop(handles)  # 맞춘 bag도 버린다
            raise
        return handles, classes

    def predict(self, handles: list[tuple[int, int]], X: Any, proba: bool) -> list[np.ndarray]:
        """맞춘 bag마다 `X`의 예측(`proba`면 predict_proba)."""
        n = len(self.workers)
        if any(w != i % n for i, (w, _) in enumerate(handles)):
            raise ValueError("handle은 fit이 돌려준 순서 그대로 준다")
        return self._rounds([("predict", key, X, proba) for _, key in handles])

    def drop(self, handles: list[tuple[int, int]]) -> None:
        by_worker: dict[int, list[int]] = {}
        for w, key in handles:
            by_worker.setdefault(w, []).append(key)
        for w, keys in by_worker.items():
            self._send(w, ("drop", keys))

    def close(self) -> None:
        for _, conn in self.workers:
            try:
                conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for process, conn in self.workers:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
            conn.close()
        self.workers = []
