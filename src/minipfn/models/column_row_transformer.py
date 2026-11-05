"""Column-Row Transformer — 열 → 행 → 행 ICL 세 단계로 표를 읽는 Transformer (TabPFN-3 Technical
Report arXiv 2605.13986 Fig. 5, Appendix C 표 2~7의 구조. TabICL 계열의 뼈대).

`cell_transformer`(v2·2.5)는 셀 묶음 토큰 하나하나가 층마다 행·열 두 방향으로 attention한다.
여기서는 단계마다 한 방향만 본다.

1. 셀 encoder: feature마다 학습 행으로 z-정규화하고(결측 칸은 0, 표시를 따로), feature j의 토큰은
   circular shift로 묶은 셀 3개 (j, j+1, j+3 mod C)와 결측 표시 3개 → Linear → E(128). 학습 행에는
   정답 embedding을 더한다(분류: 클래스마다 직교 초기화한 벡터, 회귀: Linear(y))
2. 열 단계 (`column_blocks`개): 열마다 inducing point `n_inducing`(128)개가 학습 행에
   cross-attention해 열의 분포를 요약하고, 모든 행(학습·테스트)이 그 요약에 attention한다
   (ISAB). 테스트 행끼리는 서로 보지 않는다
3. 행 단계 (`row_blocks`개): 행마다 학습되는 CLS 토큰 `n_cls`(4)개와 그 행의 feature 토큰이 서로
   attention한다(RoPE, base 1e5). CLS 출력을 이어 붙여 행 벡터 하나(n_cls × E = 512)
4. 행 ICL (`nlayers`개, 24): 학습 행은 학습 행끼리, 테스트 행은 학습 행에만 attention한다. 테스트
   행은 key·value head 하나만 쓴다(multi-query)
5. 출력: 분류는 many-class retrieval decoder — 테스트 행(query)이 학습 행(key)에 attention한
   가중치로 학습 행의 one-hot 정답을 평균해 확률 p를 내고 log(clip(p))를 logit으로 둔다
   (head 6개 × 64, head 평균). 클래스 수에 따로 파라미터가 없고 상한은 정답 embedding
   수(`n_out`)다. 회귀는 MLP (D → 2D → GELU → 구간 5,000개)

모든 normalization은 RMSNorm이고 ff_factor는 2다. 분류 53M·회귀 58M (TabPFN-3 기본 크기).

보고서에 없어서 여기서 정한 것:

- QASSMax(query-aware scalable softmax): 보고서는 "입력 길이에 따라 query를 다시 스케일한다",
  "hidden 64인 MLP"만 적었다. 여기서는 SSMax(Nakanishi 2025)의 log n 스케일에 query마다 MLP가 더하는
  항을 붙여 q ← q · log(n) · (s_h + MLP(q_h))로 둔다(n은 key 수, s_h는 head마다 학습, MLP 마지막
  층은 0으로 시작해 처음에는 SSMax다). 열 단계·행 ICL·retrieval decoder에 쓴다(보고서대로)
- RMSNorm 위치: pre-norm(attention·MLP 앞), 모든 단계 마지막에도 하나. 학습 파라미터 없는
  RMSNorm이다
- QK-clip(`training.stability`)은 쓰지 않는다(`qk_clip_weights`가 빈 목록). QK-clip은 head마다
  σ(Wq_hᵀ Wk_h)로 logit 상한을 잡는데, 여기서는 그 상한이 실제 logit을 묶지 못한다: QASSMax가
  q에 학습되는 배율 log(n) · (s_h + MLP(q))를 곱하고(가중치로 묶이지 않는다), 테스트 행은
  multi-query라 head h의 query가 head 0의 key를 본다(q_h · k_0). logit 폭주를 막는 것은 3.5의
  `qk_norm`이다
- 정답 embedding 위치: 보고서대로 학습 행의 셀 토큰(feature마다)에 더한다. 행 ICL 입력에는 따로
  더하지 않는다
- 묶음 offset: 2^j − 1 (j = 0..G−1). G=3이면 보고서의 0, 1, 3이다

TabPFN-3.5(arXiv 2609.17895)가 바꾼 것은 설정으로 켠다 (recipe `ARCHITECTURES["v3.5"]`).

- 폭: `n_cls` 4 → 8 (행 벡터 512 → 1024), `nhead` 8 → 16 (head dim 64)
- `qk_norm`: attention의 q·k에 RMSNorm(head dim마다 학습되는 gain, 여기서 정함)
- `encoder="fourier"`: 셀 encoder. z값에 학습되는 주파수 32개를 곱한 sin·cos(묶음 위치마다)를
  Linear한 것과, [z값, 결측 표시, 열 안 ECDF midrank u의 sin·cos(πku), k = 1..4] 묶음을 Linear한
  것을 더해 LayerNorm. 주파수의 처음 값(log 간격 0.1~10)은 여기서 정함. 학습 행에서 상수인 열은
  z처럼 u도 모든 행에서 0.5로 둔다(테스트 행에만 있는 신호가 되지 않도록)
- 분류·회귀를 한 checkpoint로(task를 입력으로) 학습하는 것은 아직 없다. 이 클래스는 `task` 하나만
  가진다
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from minipfn.models.cell_transformer import standardize_cells

TASKS = ("classification", "regression")
ENCODERS = ("linear", "fourier")
N_FOURIER = 32  # TabPFN-3.5 cell encoder의 주파수 수
N_HARMONICS = 4  # TabPFN-3.5 ECDF의 sin·cos 차수
CLIP_PROB = 1e-6  # retrieval decoder의 log(clip(p))


@dataclass
class ColumnRowTransformerConfig:
    """모델 크기. 기본값은 TabPFN-3 기본 모델(Appendix C 표 2~7)."""

    task: str = "classification"  # 출력: 분류 retrieval decoder 또는 회귀 MLP
    emsize: int = 128  # 셀·열·행 단계의 폭 E
    features_per_group: int = 3
    n_inducing: int = 128
    column_blocks: int = 3
    column_heads: int = 8
    n_cls: int = 4  # 행 ICL 폭 = n_cls × emsize
    row_blocks: int = 3
    row_heads: int = 8
    rope_base: float = 100_000.0
    nlayers: int = 24  # 행 ICL 층 수
    nhead: int = 8  # 행 ICL head 수
    ff_factor: int = 2
    decoder_heads: int = 6  # 분류 retrieval decoder
    decoder_head_dim: int = 64
    qassmax_hidden: int = 64
    qk_norm: bool = False  # TabPFN-3.5: True
    encoder: str = "linear"  # 셀 encoder. TabPFN-3.5: "fourier"
    n_out: int = 10  # 분류: 정답 embedding 수(클래스 상한), 회귀: bar distribution 구간 수
    # 학습 때 본 feature 수 상한 (`TrainConfig.max_features`). 구조와 상관없고 추론이 읽는다
    max_features: int = 160

    INFO_FIELDS: ClassVar[tuple[str, ...]] = ("max_features",)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task는 {TASKS} 가운데 하나여야 한다: {self.task}")
        if self.encoder not in ENCODERS:
            raise ValueError(f"encoder는 {ENCODERS} 가운데 하나여야 한다: {self.encoder}")
        for name, heads, dim in (
            ("column_heads", self.column_heads, self.emsize),
            ("row_heads", self.row_heads, self.emsize),
            ("nhead", self.nhead, self.icl_dim),
        ):
            if dim % heads:
                raise ValueError(f"{name} {heads}가 폭 {dim}을 나누지 않는다")
        if (self.emsize // self.row_heads) % 2:
            raise ValueError("RoPE를 쓰려면 행 단계 head dim이 짝수여야 한다")

    @property
    def icl_dim(self) -> int:
        return self.n_cls * self.emsize

    def to_dict(self) -> dict:
        return asdict(self)


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """학습 파라미터 없는 RMSNorm (마지막 축)."""
    return F.rms_norm(x, (x.shape[-1],))


def group_offsets(g: int) -> tuple[int, ...]:
    """feature j의 토큰에 묶는 열의 offset: 2^k − 1 (k = 0..g−1). g=3이면 (0, 1, 3)."""
    return tuple(2**k - 1 for k in range(g))


def _group(t: torch.Tensor, offsets: tuple[int, ...]) -> torch.Tensor:
    # (B, R, C) → (B, R, C, G): feature j 자리에 열 (j + offset) mod C
    c = t.shape[-1]
    idx = (torch.arange(c, device=t.device)[:, None] + torch.tensor(offsets, device=t.device)) % c
    return t[..., idx]


def ecdf_midrank(x: torch.Tensor, n_train: int) -> torch.Tensor:
    """열마다 학습 행에 대한 midrank u ∈ [0, 1]: (작은 것 수 + 같은 것 수 / 2) / 결측 아닌 학습
    행 수. 결측 칸은 0.5. x: (B, R, C)."""
    finite = torch.isfinite(x)
    train = torch.where(finite[:, :n_train], x[:, :n_train], torch.inf)
    sorted_train = train.transpose(1, 2).sort(-1).values.contiguous()  # (B, C, N), 결측은 뒤로
    values = torch.where(finite, x, 0.0).transpose(1, 2).contiguous()  # (B, C, R)
    left = torch.searchsorted(sorted_train, values, side="left")
    right = torch.searchsorted(sorted_train, values, side="right")
    n_valid = finite[:, :n_train].sum(1).clamp_min(1)  # (B, C)
    u = ((left + right).to(x.dtype) / 2 / n_valid[..., None].to(x.dtype)).clamp(0, 1)
    return torch.where(finite, u.transpose(1, 2), 0.5)


def _rope(t: torch.Tensor, base: float) -> torch.Tensor:
    # t: (N, H, L, d). 위치 0..L-1로 앞 절반·뒤 절반을 회전 (rotate-half)
    length, d = t.shape[-2], t.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, device=t.device, dtype=torch.float32) / d)
    angle = torch.arange(length, device=t.device, dtype=torch.float32)[:, None] * inv
    cos, sin = angle.cos().to(t.dtype), angle.sin().to(t.dtype)
    a, b = t[..., : d // 2], t[..., d // 2 :]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class Attention(nn.Module):
    """multi-head attention. q·k·v를 한 weight(`qkv`, (3D, D))로 둔다.

    - `context`를 주면 key·value는 거기서(cross-attention), 아니면 query와 같은 입력에서
    - `n_context`를 주면(self-attention) 앞의 n_context개만 key·value가 되고, 그 뒤 query(테스트
      행)는 첫 head의 key·value만 본다(multi-query)
    - `qassmax`: q ← q · log(key 수) · (s_h + MLP(q_h)) (모듈 docstring)
    - `qk_norm`: q·k에 RMSNorm (TabPFN-3.5), `rope_base`: q·k에 RoPE
    - `zero_out`: 출력층을 0으로 시작한다 (처음에는 잔차에 아무것도 더하지 않는다)
    """

    def __init__(
        self,
        dim: int,
        nhead: int,
        *,
        qassmax: bool = False,
        qassmax_hidden: int = 64,
        qk_norm: bool = False,
        rope_base: float | None = None,
        zero_out: bool = True,
    ) -> None:
        super().__init__()
        self.nhead, self.qk_norm, self.rope_base = nhead, qk_norm, rope_base
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        if qk_norm:
            # 학습되는 gain이 있는 RMSNorm. gain이 없으면 logit이 √(head dim) 안에 묶여(행 단계
            # head dim 16이면 4) attention이 거의 고르게 된다
            self.q_norm = nn.RMSNorm(dim // nhead)
            self.k_norm = nn.RMSNorm(dim // nhead)
        if zero_out:
            nn.init.zeros_(self.out.weight)  # 처음에는 잔차에 아무것도 더하지 않는다
        self.qassmax = QASSMax(nhead, dim // nhead, qassmax_hidden) if qassmax else None

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        n, length, dim = t.shape
        return t.view(n, length, self.nhead, dim // self.nhead).transpose(1, 2)  # (N, H, L, d)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        n_context: int | None = None,
    ) -> torch.Tensor:
        n, length, dim = x.shape
        w = self.qkv.weight
        source = x if context is None else context
        if n_context is not None:
            source = source[:, :n_context]
        q = self._heads(F.linear(x, w[:dim]))
        k = self._heads(F.linear(source, w[dim : 2 * dim]))
        v = self._heads(F.linear(source, w[2 * dim :]))
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope_base is not None:
            q, k = _rope(q, self.rope_base), _rope(k, self.rope_base)
        if self.qassmax is not None:
            q = self.qassmax(q, k.shape[2])
        if n_context is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            train = F.scaled_dot_product_attention(q[:, :, :n_context], k, v)
            # 테스트 행: 모든 query head가 첫 head의 key·value를 본다. head를 query 길이로 접어
            # key·value head 하나로 계산한다(key·value를 head 수만큼 늘리지 않도록)
            test_q = q[:, :, n_context:]
            h, t, d = test_q.shape[1:]
            test = F.scaled_dot_product_attention(
                test_q.reshape(n, 1, h * t, d), k[:, :1], v[:, :1]
            ).view(n, h, t, d)
            out = torch.cat([train, test], dim=2)
        return self.out(out.transpose(1, 2).reshape(n, length, dim))


class QASSMax(nn.Module):
    """query-aware scalable softmax: q ← q · log(key 수) · (s_h + MLP(q_h)) (모듈 docstring).
    처음 배율은 log(n) / log(256)이라 key 256개일 때 보통 attention과 같다 (여기서 정함)."""

    def __init__(self, nhead: int, head_dim: int, hidden: int) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.full((nhead,), 1 / math.log(256)))
        self.mlp = nn.Sequential(nn.Linear(head_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[2].weight)  # 처음에는 query와 상관없는 SSMax
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, q: torch.Tensor, n_keys: int) -> torch.Tensor:
        # q: (N, H, L, d)
        scale = self.base.to(q.dtype)[None, :, None, None] + self.mlp(q)
        return q * scale * math.log(max(n_keys, 2))


def _mlp(dim: int, ff_factor: int) -> nn.Sequential:
    mlp = nn.Sequential(
        nn.Linear(dim, ff_factor * dim, bias=False),
        nn.GELU(),
        nn.Linear(ff_factor * dim, dim, bias=False),
    )
    nn.init.zeros_(mlp[2].weight)
    return mlp


class Block(nn.Module):
    """pre-norm self-attention + MLP (행 단계, 행 ICL)."""

    def __init__(self, dim: int, nhead: int, ff_factor: int, **attention) -> None:
        super().__init__()
        self.attention = Attention(dim, nhead, **attention)
        self.mlp = _mlp(dim, ff_factor)

    def forward(self, h: torch.Tensor, n_context: int | None = None) -> torch.Tensor:
        h = h + self.attention(rms_norm(h), n_context=n_context)
        return h + self.mlp(rms_norm(h))


class InducedBlock(nn.Module):
    """열 단계 블록 하나 (ISAB): inducing point가 학습 행을 요약하고, 모든 행이 그 요약을 본다."""

    def __init__(self, dim: int, nhead: int, n_inducing: int, ff_factor: int, **attention) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(n_inducing, dim))
        self.summarize = Attention(dim, nhead, **attention)
        self.summarize_mlp = _mlp(dim, ff_factor)
        self.broadcast = Attention(dim, nhead, **attention)
        self.broadcast_mlp = _mlp(dim, ff_factor)

    def forward(self, h: torch.Tensor, n_train: int) -> torch.Tensor:
        # h: (N, R, E), N = 데이터셋 × 열. 앞의 n_train행이 학습 행
        ind = self.inducing.to(h.dtype).expand(h.shape[0], -1, -1)
        ind = ind + self.summarize(rms_norm(ind), context=rms_norm(h[:, :n_train]))
        ind = ind + self.summarize_mlp(rms_norm(ind))
        h = h + self.broadcast(rms_norm(h), context=rms_norm(ind))
        return h + self.broadcast_mlp(rms_norm(h))


class FourierCellEncoder(nn.Module):
    """TabPFN-3.5 셀 encoder: 학습 주파수 sin·cos 항 + [z값, 결측 표시, ECDF 조화 항] 항
    → LayerNorm."""

    def __init__(self, g: int, dim: int) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.logspace(-1, 1, N_FOURIER))
        self.fourier = nn.Linear(g * 2 * N_FOURIER, dim)
        self.metadata = nn.Linear(g * (2 + 2 * N_HARMONICS), dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, indicator: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # 각각 (B, R, C, G)
        b, r, c, g = z.shape
        phase = z[..., None] * self.frequencies.to(z.dtype)  # (B, R, C, G, F)
        fourier = torch.cat([phase.sin(), phase.cos()], dim=-1).reshape(b, r, c, -1)
        k = torch.arange(1, N_HARMONICS + 1, device=u.device, dtype=u.dtype)
        harmonics = math.pi * u[..., None] * k
        meta = torch.cat(
            [z[..., None], indicator[..., None], harmonics.sin(), harmonics.cos()], dim=-1
        ).reshape(b, r, c, -1)
        return self.norm(self.fourier(fourier) + self.metadata(meta))


class ColumnRowTransformer(nn.Module):
    # 모델 종류 (`minipfn.models.MODEL_TYPES`)
    model_type = "column_row_transformer"
    config_class = ColumnRowTransformerConfig

    def __init__(self, config: ColumnRowTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or ColumnRowTransformerConfig()
        e, g, d = config.emsize, config.features_per_group, config.icl_dim
        self.offsets = group_offsets(g)
        if config.encoder == "fourier":
            self.cell_encoder = FourierCellEncoder(g, e)
        else:
            self.cell_encoder = nn.Linear(2 * g, e)  # [값 g개, 결측 표시 g개] → 토큰
        if config.task == "classification":
            self.target_embedder = nn.Embedding(config.n_out, e)
            nn.init.orthogonal_(self.target_embedder.weight)
        else:
            self.target_embedder = nn.Linear(1, e)
        qassmax = {"qassmax": True, "qassmax_hidden": config.qassmax_hidden}
        self.column_blocks = nn.ModuleList(
            InducedBlock(
                e,
                config.column_heads,
                config.n_inducing,
                config.ff_factor,
                qk_norm=config.qk_norm,
                **qassmax,
            )
            for _ in range(config.column_blocks)
        )
        # 작게 시작한다 (여기서 정함): 크기가 1이면 행 벡터가 CLS 값에 묻혀 처음에 모든 행이 거의
        # 같다(1 − cos ≈ 0.015). 작으면 CLS 잔차가 attention이 모은 셀 정보로 채워진다
        self.cls_tokens = nn.Parameter(0.02 * torch.randn(config.n_cls, e))
        self.row_blocks = nn.ModuleList(
            # CLS 토큰은 attention으로만 행의 셀 정보를 받는다. 출력층이 0에서 시작하면 처음에는
            # 모든 행 벡터가 같아 gradient가 거의 0이라(1e-8) 학습이 시작되지 않는다
            Block(
                e,
                config.row_heads,
                config.ff_factor,
                qk_norm=config.qk_norm,
                rope_base=config.rope_base,
                zero_out=False,
            )
            for _ in range(config.row_blocks)
        )
        self.icl_layers = nn.ModuleList(
            Block(d, config.nhead, config.ff_factor, qk_norm=config.qk_norm, **qassmax)
            for _ in range(config.nlayers)
        )
        if config.task == "classification":
            dh = config.decoder_heads * config.decoder_head_dim
            self.decoder = nn.Linear(d, 2 * dh, bias=False)  # [W_Q; W_K]
            self.decoder_qassmax = QASSMax(
                config.decoder_heads, config.decoder_head_dim, config.qassmax_hidden
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(d, config.ff_factor * d),
                nn.GELU(),
                nn.Linear(config.ff_factor * d, config.n_out),
            )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def qk_clip_weights(self) -> list[tuple[str, torch.Tensor]]:
        """빈 목록: QK-clip의 가중치 상한이 이 모델의 logit을 묶지 못한다 (QASSMax의 학습 배율,
        테스트 행의 multi-query. 모듈 docstring). 상한을 지표로 남기면 실제보다 안전해 보이므로
        지표도 내지 않는다."""
        return []

    def decoder_weight(self) -> torch.Tensor:
        return (
            self.decoder.weight if self.config.task == "classification" else self.decoder[-1].weight
        )

    def _embed_cells(self, x: torch.Tensor, n_train: int) -> torch.Tensor:
        # x: (B, R, C) → 토큰 (B, R, C, E). feature j의 토큰은 열 j + offset들의 셀
        z, indicator, used = standardize_cells(x, n_train)
        zg, ig = _group(z, self.offsets), _group(indicator, self.offsets)
        if self.config.encoder == "fourier":
            u = ecdf_midrank(x.float(), n_train).to(z.dtype)
            # 학습 행에서 상수인 열: z를 0으로 둔 것처럼 u도 모든 행에서 0.5 (학습 행은 모두 0.5이고
            # 테스트 행만 0이나 1이 되어 테스트 행에만 있는 신호가 되지 않도록)
            u = torch.where(used > 0, u, 0.5)
            return self.cell_encoder(zg, ig, _group(u, self.offsets))
        return self.cell_encoder(torch.cat([zg, ig], dim=-1))

    def _embed_target(self, y_train: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # (B, n_train) → (B, n_train, E)
        if self.config.task == "classification":
            classes = y_train.long().clamp(0, self.config.n_out - 1)
            return self.target_embedder(classes).to(like.dtype)
        return self.target_embedder(y_train.to(like.dtype)[..., None])

    def _retrieve(self, h: torch.Tensor, y_train: torch.Tensor, n_train: int) -> torch.Tensor:
        # 분류 retrieval decoder: (B, R, D) → 테스트 행 logit (B, T, n_out)
        b = h.shape[0]
        heads, dh = self.config.decoder_heads, self.config.decoder_head_dim
        # query는 테스트 행에서만, key는 학습 행에서만 만든다 (weight는 [W_Q; W_K] 하나)
        w_q, w_k = self.decoder.weight.split(heads * dh)
        q = F.linear(h[:, n_train:], w_q).reshape(b, -1, heads, dh).transpose(1, 2)  # (B, H, T, dh)
        k = F.linear(h[:, :n_train], w_k).reshape(b, -1, heads, dh).transpose(1, 2)  # (B, H, N, dh)
        q = self.decoder_qassmax(q, n_train)
        # 확률·log는 float32로 (bf16이면 clip 값 근처가 뭉개진다)
        weights = torch.softmax((q @ k.transpose(-1, -2)).float() / math.sqrt(dh), dim=-1)
        onehot = F.one_hot(y_train.long().clamp(0, self.config.n_out - 1), self.config.n_out)
        p = (weights @ onehot.float()[:, None]).mean(1)  # (B, T, n_out): head 평균
        return p.clamp_min(CLIP_PROB).log()

    def forward(self, x: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        """테스트 행의 logit (B, R - n_train, n_out).

        x: (B, R, C), 학습 행 다음에 테스트 행. y_train: (B, n_train).
        분류면 y는 0부터 시작하는 클래스 번호, 회귀면 z-정규화한 값이다.
        """
        cfg = self.config
        b, r, c = x.shape
        n_train = y_train.shape[1]
        cells = self._embed_cells(x, n_train)  # (B, R, C, E)
        target = self._embed_target(y_train, cells)  # (B, N, E)
        target = F.pad(target, (0, 0, 0, r - n_train))  # 테스트 행은 0
        cells = cells + target[:, :, None]
        e = cells.shape[-1]

        # 1) 열 단계: (B × C, R, E)
        h = cells.permute(0, 2, 1, 3).reshape(b * c, r, e)
        for block in self.column_blocks:
            h = block(h, n_train)

        # 2) 행 단계: (B × R, n_cls + C, E) → CLS 출력을 이어 붙인 행 벡터 (B, R, n_cls × E)
        h = h.view(b, c, r, e).transpose(1, 2).reshape(b * r, c, e)
        h = torch.cat([self.cls_tokens.to(h.dtype).expand(b * r, -1, -1), h], dim=1)
        for block in self.row_blocks:
            h = block(h)
        rows = rms_norm(h[:, : cfg.n_cls]).reshape(b, r, cfg.icl_dim)

        # 3) 행 ICL
        for layer in self.icl_layers:
            rows = layer(rows, n_context=n_train)
        rows = rms_norm(rows)

        # 4) 출력
        if cfg.task == "classification":
            return self._retrieve(rows, y_train, n_train)
        return self.decoder(rows[:, n_train:])
