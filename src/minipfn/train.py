"""Prior-fitting 학습 루프 — 논문 Algorithm 1을 그대로 반복한다.

매 스텝: GP prior에서 데이터셋 batch를 뽑고 → context/query로 나눠 → query NLL을 계산하고
→ Adam 스텝. 실제 데이터는 쓰지 않는다(prior가 곧 학습 분포). 학습된 모델은 forward pass
한 번으로 새 데이터셋의 예측분포를 낸다(in-context learning).
"""

from __future__ import annotations

import copy
import importlib
import math
import pkgutil
import sys
from dataclasses import asdict, dataclass

import torch
from torch import nn

from minipfn.bar_distribution import FullSupportBarDistribution, get_bucket_borders
from minipfn.gp_prior import (
    DEFAULT_LENGTHSCALE,
    DEFAULT_NOISE,
    DEFAULT_VARIANCE,
    GPPriorSampler,
)
from minipfn.model import PFNTransformer


@dataclass
class PFNConfig:
    """모델·학습·GP prior 설정. 로컬 디버깅용 small(), A10용 large()."""

    # 모델
    emsize: int = 32
    nhead: int = 2
    nlayers: int = 2
    nhid: int = 128
    num_buckets: int = 100
    dropout: float = 0.0
    # bar distribution / GP prior
    bucket_range: tuple[float, float] = (-4.0, 4.0)
    x_range: tuple[float, float] = (-5.0, 5.0)
    lengthscale: float = DEFAULT_LENGTHSCALE
    variance: float = DEFAULT_VARIANCE
    noise: float = DEFAULT_NOISE
    # 데이터셋 구성
    seq_len: int = 60
    min_context: int = 5
    max_context: int = 50
    # 최적화
    batch_size: int = 16
    steps: int = 1200
    lr: float = 1e-3
    warmup_steps: int = 0
    seed: int = 42

    @classmethod
    def small(cls) -> PFNConfig:
        return cls()

    @classmethod
    def large(cls) -> PFNConfig:
        return cls(
            emsize=256,
            nhead=4,
            nlayers=6,
            nhid=1024,
            num_buckets=1000,
            seq_len=200,
            min_context=5,
            max_context=190,
            batch_size=128,
            steps=30000,
            lr=3e-4,
            warmup_steps=1000,
        )

    @classmethod
    def paper(cls) -> PFNConfig:
        """논문(Müller et al. 2021) GP-fit 설정에 맞춘 config.

        원 저장소 train.py / 부록 기준으로 모델 크기와 step 수를 논문값으로 맞춘다:
        emsize 512, 6 layers, 4 heads, nhid=2*emsize=1024, dropout 0, ~10,000 step
        (100 epoch × 100 step). batch/seq/lr/버킷은 논문이 명시하지 않아 우리 기본값을 쓴다.
        A10 기준 대략 2~3시간 소요.
        """
        return cls(
            emsize=512,
            nhead=4,
            nlayers=6,
            nhid=1024,
            num_buckets=100,
            seq_len=200,
            min_context=5,
            max_context=190,
            batch_size=128,
            steps=10000,
            lr=3e-4,
            warmup_steps=1000,
        )


def build_criterion(config: PFNConfig) -> FullSupportBarDistribution:
    borders = get_bucket_borders(config.num_buckets, config.bucket_range)
    return FullSupportBarDistribution(borders)


def build_model(config: PFNConfig) -> PFNTransformer:
    return PFNTransformer(
        x_dim=1,
        emsize=config.emsize,
        nhead=config.nhead,
        nhid=config.nhid,
        nlayers=config.nlayers,
        num_buckets=config.num_buckets,
        dropout=config.dropout,
    )


def build_prior_sampler(config: PFNConfig) -> GPPriorSampler:
    return GPPriorSampler(
        x_range=config.x_range,
        lengthscale=config.lengthscale,
        variance=config.variance,
        noise=config.noise,
    )


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if warmup >= total:
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _log_best_and_final_models(mlf, model, best_state, best_val, best_step, config) -> None:
    """best/final 모델을 MLflow(mlflow.pytorch)로 로깅한다. config는 metadata로 동봉해

    나중에 criterion을 재구성할 수 있게 한다."""
    import mlflow.pytorch

    meta = {"pfn_config": asdict(config)}
    # pickle 포맷: 커스텀 attention-mask forward를 트레이싱(pt2)할 필요가 없어 안정적.
    model.eval()
    mlflow.pytorch.log_model(
        model, name="final_model", metadata=meta, serialization_format="pickle"
    )
    if best_state is not None:
        best_model = copy.deepcopy(model)
        best_model.load_state_dict(best_state)
        best_model.eval()
        mlflow.pytorch.log_model(
            best_model, name="best_model", metadata=meta, serialization_format="pickle"
        )
        mlf.log_metric("best_val_nll", best_val)
        mlf.log_param("best_step", best_step)
    model.train()


def train_pfn(
    model: nn.Module,
    prior_sampler: GPPriorSampler,
    criterion: nn.Module,
    config: PFNConfig,
    device: torch.device | str = "cpu",
    log_every: int = 50,
    progress: bool = True,
    mlflow_log: bool = False,
    run_name: str | None = None,
    metric_every: int = 10,
    eval_every: int | None = None,
    log_models: bool = True,
) -> dict[str, list[float]]:
    """prior-fitting 루프. loss history를 반환한다.

    `mlflow_log=True`면 config를 params로, step별 train NLL을 metric으로 기록한다
    (`metric_every` 스텝마다). 또 `eval_every`(기본 `steps//10`) 스텝마다 held-out prior
    batch의 검증 NLL을 재어 `val_nll`로 기록하고, 개선될 때의 가중치를 best로 추적한다.
    끝나면 `log_models=True`일 때 best/final 모델을 MLflow(mlflow.pytorch)로 로깅한다.
    이미 활성 run이 있으면 거기에, 없으면 새 run을 연다(Databricks는 노트북 experiment로 연결).
    """
    device = torch.device(device)
    model.to(device)
    criterion.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, config.warmup_steps, config.steps)
    )

    mlf = None
    started_run = False
    val_batch = None
    best_val, best_state, best_step = math.inf, None, -1
    if mlflow_log:
        import mlflow as mlf

        if mlf.active_run() is None:
            mlf.start_run(run_name=run_name)
            started_run = True
        mlf.log_params(asdict(config))
        mlf.log_param("device", str(device))
        mlf.log_param("n_parameters", sum(p.numel() for p in model.parameters()))
        if eval_every is None:
            eval_every = max(1, config.steps // 10)
        # held-out 검증 batch — prior에서 새로 뽑으므로 학습이 이 batch를 다시 볼 확률은 0.
        xv, yv = prior_sampler.sample_batch(config.batch_size, config.seq_len, device)
        sep_val = (config.min_context + config.max_context) // 2
        val_batch = (xv[:, :sep_val], yv[:, :sep_val], xv[:, sep_val:], yv[:, sep_val:])

    def _val_nll() -> float:
        model.eval()
        with torch.no_grad():
            xc, yc, xq, yq = val_batch
            vl = criterion(model(xc, yc, xq), yq).mean().item()
        model.train()
        return vl

    losses: list[float] = []
    step_iter = range(config.steps)
    if progress:
        try:
            from tqdm.auto import tqdm

            step_iter = tqdm(step_iter, desc="train")
        except ImportError:
            pass

    try:
        for step in step_iter:
            sep = int(torch.randint(config.min_context, config.max_context + 1, (1,)).item())
            x, y = prior_sampler.sample_batch(config.batch_size, config.seq_len, device)
            x_ctx, y_ctx = x[:, :sep], y[:, :sep]
            x_qry, y_qry = x[:, sep:], y[:, sep:]

            logits = model(x_ctx, y_ctx, x_qry)
            loss = criterion(logits, y_qry).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            losses.append(loss.item())
            if progress and hasattr(step_iter, "set_postfix") and step % log_every == 0:
                step_iter.set_postfix(loss=f"{loss.item():.4f}")
            if mlf is not None and step % metric_every == 0:
                mlf.log_metric("train_nll", loss.item(), step=step)
                mlf.log_metric("lr", scheduler.get_last_lr()[0], step=step)
            if mlf is not None and step % eval_every == 0:
                vl = _val_nll()
                mlf.log_metric("val_nll", vl, step=step)
                if vl < best_val:
                    best_val, best_state, best_step = vl, copy.deepcopy(model.state_dict()), step

        if mlf is not None:
            last = config.steps - 1
            mlf.log_metric("train_nll", losses[-1], step=last)
            tail = losses[-50:]
            mlf.log_metric("final_train_nll", sum(tail) / len(tail))
            vl = _val_nll()
            mlf.log_metric("val_nll", vl, step=last)
            if vl < best_val:
                best_val, best_state, best_step = vl, copy.deepcopy(model.state_dict()), last
            if log_models:
                _log_best_and_final_models(mlf, model, best_state, best_val, best_step, config)
    finally:
        if started_run:
            mlf.end_run()

    return {"losses": losses, "best_val_nll": best_val, "best_step": best_step}


@torch.no_grad()
def pfn_predict(
    model: nn.Module,
    criterion: FullSupportBarDistribution,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
    center_prob: float = 0.682,
) -> dict[str, torch.Tensor]:
    """in-context 예측. x_context (nc, 1), y_context (nc,), x_query (nq, 1).

    반환: mean/std/lower/upper (각 (nq,)), logits (nq, num_buckets).
    """
    model.eval()
    device = next(model.parameters()).device
    xc = x_context.to(device).reshape(1, -1, 1)
    yc = y_context.to(device).reshape(1, -1)
    xq = x_query.to(device).reshape(1, -1, 1)

    logits = model(xc, yc, xq)[0]  # (nq, num_buckets)
    mean = criterion.mean(logits)
    std = criterion.std(logits)
    lower, upper = criterion.quantile(logits, center_prob=center_prob)
    return {"mean": mean, "std": std, "lower": lower, "upper": upper, "logits": logits}


def save_checkpoint(model: nn.Module, config: PFNConfig, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict:
    """체크포인트를 읽어 {"model", "config", "criterion"}를 반환한다."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    config = PFNConfig(**ckpt["config"])
    model = build_model(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(map_location)
    criterion = build_criterion(config).to(map_location)
    return {"model": model, "config": config, "criterion": criterion}


LEGACY_PACKAGE = "pfn_cookbook"  # 2026-10 minipfn으로 이름을 바꾸기 전의 패키지 이름


def alias_legacy_package() -> None:
    """`pfn_cookbook.*`을 `minipfn.*`의 별칭으로 등록한다.

    이름을 바꾸기 전에 pickle 형식으로 로깅한 모델은 `pfn_cookbook.model.PFNTransformer`처럼
    옛 모듈 경로를 담고 있어, 별칭이 없으면 unpickle할 때 ModuleNotFoundError가 난다."""
    import minipfn

    sys.modules.setdefault(LEGACY_PACKAGE, minipfn)
    for info in pkgutil.iter_modules(minipfn.__path__):
        if not info.ispkg:
            module = importlib.import_module(f"minipfn.{info.name}")
            sys.modules.setdefault(f"{LEGACY_PACKAGE}.{info.name}", module)


def load_logged_model(model_uri: str, map_location: str | torch.device = "cpu") -> dict:
    """MLflow(mlflow.pytorch)에 로깅된 PFN 모델을 불러오고, metadata의 config로 criterion을

    재구성해 {"model", "config", "criterion"}를 반환한다. 추론에 그대로 pfn_predict에 쓴다."""
    import mlflow.pytorch
    from mlflow.models import Model

    alias_legacy_package()
    model = mlflow.pytorch.load_model(model_uri, map_location=map_location)
    meta = dict((Model.load(model_uri).metadata or {}).get("pfn_config", {}))
    for key in ("bucket_range", "x_range"):
        if meta.get(key) is not None:
            meta[key] = tuple(meta[key])
    config = PFNConfig(**meta) if meta else None
    criterion = build_criterion(config).to(map_location) if config is not None else None
    return {"model": model.to(map_location).eval(), "config": config, "criterion": criterion}
