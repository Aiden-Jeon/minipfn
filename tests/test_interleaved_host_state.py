"""`stage_layout="interleaved"` 추론의 CPU 셀 상태 (`column_row_transformer._host_state`): caching
host allocator(`pin_memory=True`) 대신 정확한 크기의 pageable 텐서를 잡아 그 자리만 pinned로
등록하고, forward가 끝나면(예외여도)푼다. 값은 예전(상태를 장치에 둘 때)과 비트 동일하다."""

import warnings

import pytest
import torch

from minipfn.models import column_row_transformer as M
from minipfn.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig
from minipfn.recipes import kumo_tabular

V35 = {"qk_norm": True, "encoder": "fourier", "target_norm": True, "head_mlp": True}
TINY_STAGES = {"stage_layout": "interleaved", "column_blocks": 2, "row_blocks": 2}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA가 필요하다")


def _model(**changes) -> ColumnRowTransformer:
    torch.manual_seed(0)
    config = ColumnRowTransformerConfig.tiny(
        task="multitask", n_buckets=7, **V35, **kumo_tabular.ATTENTION_TEMPERATURE, **changes
    )
    model = ColumnRowTransformer(config).eval()
    model.inference_chunk_cells = 37  # 행 묶음 여럿, 열 묶음 여럿
    model.inference_column_chunk = 2
    return model


def _inputs(device="cpu"):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 40, 5, generator=g)
    x[0, 3, 1] = float("nan")
    yc = torch.randint(0, 3, (1, 30), generator=g).float()
    yr = torch.randn(1, 30, generator=g)
    return x.to(device), yc.to(device), yr.to(device)


def _forward(model, x, y, task):
    with torch.inference_mode():
        return model(x, y, task, inference=True)


def test_offloaded_state_matches_device_state(monkeypatch):
    # CPU에서도 offload 길(`_host_state`, CUDA가 없으면 등록 없이 pageable)을 지나 같은 값을 낸다
    model = _model(**TINY_STAGES)
    x, yc, yr = _inputs()
    want = {t: _forward(model, x, y, t) for t, y in (("classification", yc), ("regression", yr))}
    calls = []
    real = M._host_state

    def host_state(shape, dtype, pin):
        calls.append((shape, dtype, pin))
        return real(shape, dtype, pin)

    monkeypatch.setattr(M, "_host_state", host_state)
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    for task, y in (("classification", yc), ("regression", yr)):
        assert torch.equal(_forward(model, x, y, task), want[task])
    assert len(calls) == 2 and calls[0][0] == (1, 40, 5, model.config.emsize)


def test_registration_is_released_after_success_and_error(monkeypatch):
    model = _model(**TINY_STAGES)
    x, yc, _ = _inputs()
    events = []

    def host_state(shape, dtype, pin):
        state = torch.empty(shape, dtype=dtype)
        events.append(("register", state.data_ptr()))
        return state, True

    def release(state, registered):
        events.append(("release", state.data_ptr(), registered))

    monkeypatch.setattr(M, "_host_state", host_state)
    monkeypatch.setattr(M, "_release_host_state", release)
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    _forward(model, x, yc, "classification")
    assert events == [("register", events[0][1]), ("release", events[0][1], True)]

    # 행 블록에서 예외가 나도 푼다 (등록한 채 남으면 그 메모리를 OS가 돌려받지 못한다)
    events.clear()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(model.row_blocks[0], "forward", boom)
    with pytest.raises(RuntimeError, match="boom"):
        _forward(model, x, yc, "classification")
    assert events == [("register", events[0][1]), ("release", events[0][1], True)]


def _counting(monkeypatch):
    # `_host_state`·`_release_host_state` 호출을 센다 (실제 함수는 그대로 부른다)
    events = []
    real_state, real_release = M._host_state, M._release_host_state

    def host_state(shape, dtype, pin):
        state, registered = real_state(shape, dtype, pin)
        events.append(("state", tuple(shape), state.data_ptr()))
        return state, registered

    def release(state, registered):
        events.append(("release", state.data_ptr()))
        real_release(state, registered)

    monkeypatch.setattr(M, "_host_state", host_state)
    monkeypatch.setattr(M, "_release_host_state", release)
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    return events


def test_hold_host_state_reuses_one_buffer_across_forwards(monkeypatch):
    model = _model(**TINY_STAGES)
    x, yc, yr = _inputs()
    want = [_forward(model, x, yc, "classification"), _forward(model, x, yr, "regression")]
    small = _forward(model, x[:, :, :3], yc, "classification")
    events = _counting(monkeypatch)
    with model.hold_host_state():
        got = [_forward(model, x, yc, "classification"), _forward(model, x, yr, "regression")]
        # 작은 상태는 같은 버퍼의 앞부분을 쓴다 (앞 forward의 값이 남지 않는다)
        got_small = _forward(model, x[:, :, :3], yc, "classification")
        assert [e[0] for e in events] == ["state"]  # 한 번만 잡고 놓지 않는다
    assert all(torch.equal(g, w) for g, w in zip(got, want, strict=True))
    assert torch.equal(got_small, small)
    ptr = events[0][2]
    assert events == [("state", (40 * 5 * model.config.emsize,), ptr), ("release", ptr)]
    assert model._held_host_state is None


def test_hold_host_state_grows_and_releases_on_error(monkeypatch):
    model = _model(**TINY_STAGES)
    x, yc, _ = _inputs()
    events = _counting(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"), model.hold_host_state():
        _forward(model, x[:, :, :3], yc, "classification")
        _forward(model, x, yc, "classification")  # 더 큰 상태: 앞 버퍼를 놓고 새로 잡는다
        raise RuntimeError("boom")
    kinds = [e[0] for e in events]
    assert kinds == ["state", "release", "state", "release"]
    assert events[1][1] == events[0][2] and events[3][1] == events[2][2]
    assert model._held_host_state is None and not model._holding_host_state


def test_ensemble_holds_one_state_per_predict(monkeypatch):
    import numpy as np

    from minipfn.inference.ensemble import EnsembleClassifier

    model = _model(**TINY_STAGES)
    model.inference_chunk_cells = 37
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.integers(0, 3, size=40)
    clf = EnsembleClassifier(model, n_estimators=3, member_kinds=("standard",)).fit(X, y)
    want = clf.predict_proba(X[:9])
    events = _counting(monkeypatch)
    got = clf.predict_proba(X[:9])
    assert np.array_equal(got, want)
    # 멤버 3개가 버퍼 하나를 쓰고, predict가 끝나면 놓는다
    assert [e[0] for e in events] == ["state", "release"]


def test_state_stays_on_device_without_offload(monkeypatch):
    # 장치에 들어가는 상태는 `_host_state`를 부르지 않는다 (CPU 장치는 늘 그렇다)
    model = _model(**TINY_STAGES)
    x, yc, _ = _inputs()
    monkeypatch.setattr(M, "_host_state", lambda *a: pytest.fail("offload하지 않는다"))
    _forward(model, x, yc, "classification")


def test_failed_registration_falls_back_to_pageable(monkeypatch):
    cleared = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            return 2  # cudaErrorMemoryAllocation

        def cudaHostUnregister(self, ptr):
            raise AssertionError("등록하지 않은 메모리는 풀지 않는다")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(M, "_clear_cuda_error", lambda: cleared.append(True))
    with pytest.warns(RuntimeWarning, match="pinned로 등록하지 못했다"):
        state, registered = M._host_state((2, 3), torch.bfloat16, True)
    assert not registered and state.shape == (2, 3) and state.device.type == "cpu"
    assert not state.is_pinned() and cleared == [True]
    M._release_host_state(state, registered)  # 등록하지 않았으면 아무것도 하지 않는다


def test_pin_off_does_not_register(monkeypatch):
    monkeypatch.setattr(torch.cuda, "cudart", lambda: pytest.fail("등록하지 않는다"))
    state, registered = M._host_state((4, 5), torch.float32, False)
    assert not registered and state.shape == (4, 5)


def test_inference_host_state_bytes():
    model = _model(**TINY_STAGES)
    e = model.config.emsize
    model.interleaved_state_device_bytes = 1000
    model.inference_chunk_cells = 100
    assert model.inference_host_state_bytes(50, 3) == 50 * 3 * e * 4  # 셀 수 > 100, 바이트 > 1000
    assert model.inference_host_state_bytes(20, 3) == 0  # 셀 60개: 나눠 계산하지 않는다
    model.interleaved_state_device_bytes = 50 * 3 * e * 4
    assert model.inference_host_state_bytes(50, 3) == 0  # 장치에 남는다
    # 기본값: BeyondArena maps_router_eta_1m (학습 + 테스트 1,250,000행, 멤버 열 768 + 식별자 1)
    default = _model(**TINY_STAGES)
    default.inference_chunk_cells = ColumnRowTransformer.inference_chunk_cells
    assert default.inference_host_state_bytes(1_250_000, 769) == 1_250_000 * 769 * e * 4
    sequential = _model()
    sequential.inference_chunk_cells = ColumnRowTransformer.inference_chunk_cells
    assert sequential.inference_host_state_bytes(1_250_000, 769) == 0  # 상태를 들지 않는다


# --- GPU -----------------------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("pin", [True, False])
def test_gpu_offload_is_bit_identical(pin):
    model = _model(**TINY_STAGES).cuda()
    x, yc, yr = _inputs("cuda")
    for task, y in (("classification", yc), ("regression", yr)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            want = _forward(model, x, y, task)  # 상태를 장치에 둔다
            model.interleaved_state_device_bytes = 0  # 상태를 CPU에 둔다
            model.interleaved_state_pin = pin
            before = torch.cuda.host_memory_stats().get("allocated_bytes.all.peak", 0)
            got = _forward(model, x, y, task)
            after = torch.cuda.host_memory_stats().get("allocated_bytes.all.peak", 0)
            model.interleaved_state_device_bytes = (
                ColumnRowTransformer.interleaved_state_device_bytes
            )
            model.interleaved_state_pin = ColumnRowTransformer.interleaved_state_pin
        assert torch.equal(got, want)
        assert after == before  # caching host allocator를 쓰지 않는다


@cuda
def test_gpu_state_size_matches_the_estimate(monkeypatch):
    # bfloat16 autocast에서도 셀 상태는 float32다: 평가 큐의 어림(`inference_host_state_bytes`)과
    # 실제 할당이 같아야 한다
    model = _model(**TINY_STAGES).cuda()
    model.interleaved_state_device_bytes = 0
    x, yc, _ = _inputs("cuda")
    sizes = []
    real = M._host_state

    def host_state(shape, dtype, pin):
        state, registered = real(shape, dtype, pin)
        sizes.append((state.dtype, state.nbytes))
        return state, registered

    monkeypatch.setattr(M, "_host_state", host_state)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _forward(model, x, yc, "classification")
    assert sizes == [(torch.float32, model.inference_host_state_bytes(*x.shape[1:]))]


@cuda
def test_gpu_register_roundtrip_and_error_clearing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state, registered = M._host_state((1024, 1024), torch.bfloat16, True)
    assert registered and state.is_pinned()
    # 같은 자리를 다시 등록하면 실패한다: 남은 오류를 지우면 다음 kernel이 그대로 돈다
    err = int(torch.cuda.cudart().cudaHostRegister(state.data_ptr(), state.nbytes, 0))
    assert err != 0
    M._clear_cuda_error()
    torch.ones(8, device="cuda").sum().item()
    M._release_host_state(state, registered)
    assert not state.is_pinned()
