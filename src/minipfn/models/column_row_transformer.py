"""Column-Row Transformer — 열 → 행 → 행 ICL 세 단계로 표를 읽는 Transformer (TabPFN-3 Technical
Report arXiv 2605.13986 Fig. 5·Appendix C 표 2~7, TabPFN-3.5 Technical Report arXiv 2609.17895의
구조. TabICL 계열의 뼈대).

`cell_transformer`(v2·2.5)는 셀 묶음 토큰 하나하나가 층마다 행·열 두 방향으로 attention한다.
여기서는 단계마다 한 방향만 본다.

1. 셀 encoder: feature마다 학습 행으로 z-정규화하고(결측 칸은 0, 표시를 따로), feature j의 토큰은
   circular shift로 묶은 셀 3개 (j, j+1, j+3 mod C)와 결측 표시 3개 → Linear → E(128). 학습 행의
   토큰에는 정답 embedding을 더한다(분류: 클래스마다 직교 초기화한 벡터, 회귀: Linear(y))
2. 열 단계 (`column_blocks`개): 열마다 inducing point `n_inducing`(128)개가 학습 행에
   cross-attention해 열의 분포를 요약하고, 모든 행(학습·테스트)이 그 요약에 attention한다
   (ISAB). 테스트 행끼리는 서로 보지 않는다
3. 행 단계 (`row_blocks`개): 행마다 학습되는 CLS 토큰 `n_cls`(4)개와 그 행의 feature 토큰이 서로
   attention한다(RoPE, base 1e5). 마지막 블록은 CLS만 query다. CLS 출력을 이어 붙여 행 벡터
   하나(n_cls × E = 512)
4. 행 ICL (`nlayers`개, 24): 학습 행 벡터에 정답 embedding을 한 번 더 더한다(셀 단계와 다른
   encoder). 학습 행은 학습 행끼리, 테스트 행은 학습 행에만 attention하고, 테스트 행은 key·value
   head 하나만 쓴다(multi-query)
5. 출력: 분류는 many-class retrieval decoder — 테스트 행(query)이 학습 행(key)에 attention한
   가중치로 학습 행의 one-hot 정답을 평균해 확률 p를 내고 log(max(p, 1e-5) + 3e-5)를 logit으로
   둔다(head 6개 × 64, head 평균). 클래스 수에 따로 파라미터가 없고 상한은 정답 embedding
   수(`n_out`)다. 회귀는 bar distribution 구간 logit

normalization은 학습되는 gain이 있는 RMSNorm(pre-norm, 단계마다 마지막에도 하나)이고 ff_factor는
2다. 분류 53M·회귀 58M (TabPFN-3 기본 크기).

보고서에는 없고 공개된 추론 코드(PriorLabs/TabPFN v9의 `architectures/tabpfn_v3*.py`, Apache 2.0.
설계만 보고 코드·가중치는 가져오지 않았다)에서 확인한 것:

- QASSMax(query-aware scalable softmax): q ← q · base_mlp(log n) · (1 + tanh(query_mlp(q))). n은
  key 수, base_mlp는 log n → (head, head dim)마다 배율, query_mlp는 head dim → head dim이고 마지막
  층이 0으로 시작한다. 열 단계의 요약 쪽(inducing point → 학습 행), 행 ICL, retrieval decoder에
  쓴다. 행 단계와 열 단계의 펼치는 쪽(모든 행 → inducing point)에는 쓰지 않는다
- 정답 embedding 두 곳(셀 토큰, 행 ICL 입력). 행 단계 마지막 블록은 CLS만 query
- 처음 값: inducing point·CLS 토큰은 표준편차 0.02, attention·MLP 출력층은 0

여기서 정한 것:

- 행 단계 attention 출력층은 0으로 시작하지 않는다. CLS 토큰은 attention으로만 행의 셀 정보를
  받는데, 0에서 시작하면 처음에 모든 행 벡터가 같아 gradient가 거의 0이었다(1e-8)
- QK-clip(`training.stability`)은 쓰지 않는다(`qk_clip_weights`가 빈 목록). QK-clip은 head마다
  σ(Wq_hᵀ Wk_h)로 logit 상한을 잡는데, QASSMax가 q에 입력에 따라 달라지는 배율을 곱하고 테스트
  행은 head h의 query가 head 0의 key를 봐서(q_h · k_0) 그 상한이 실제 logit을 묶지 못한다.
  logit을 묶는 것은 3.5의 `qk_norm`이다
- 묶음 offset: 2^j − 1 (j = 0..G−1). G=3이면 보고서의 0, 1, 3이다

TabPFN-3.5가 바꾼 것은 설정으로 켠다 (recipe `ARCHITECTURES["v3.5"]`).

- 폭: `n_cls` 4 → 8 (행 벡터 512 → 1024), `nhead` 8 → 16 (head dim 64)
- `qk_norm`: attention의 q·k에 RMSNorm (head dim마다 학습되는 gain)
- `encoder="fourier"`: 셀 encoder. z값에 학습되는 주파수 32개를 곱한 sin·cos(묶음 위치마다)를
  Linear한 것과, [z값, 결측 표시, 열 안 ECDF midrank u의 sin·cos(πku), k = 1..4] 묶음을 Linear한
  것을 더해 LayerNorm. 주파수의 처음 값(log 간격 0.1~10)은 여기서 정함. 학습 행에서 상수인 열은
  z처럼 u도 모든 행에서 0.5로 둔다(테스트 행에만 있는 신호가 되지 않도록)
- `target_norm`: 정답 embedding 뒤 LayerNorm(셀·ICL 각각 하나, 두 task가 같이 쓴다 — 정답 신호의
  크기를 task와 상관없이 맞춘다)
- `head_mlp`: 출력층 앞에 task마다 잔차 MLP 하나(pre-norm, 마지막 층 0 시작). 그 뒤 회귀 출력층은
  Linear 하나(D → 구간 수)
- `task="multitask"`: 분류·회귀를 한 모델로. 정답 encoder(셀·ICL)와 출력층(앞 MLP 포함)만 task마다
  있고 나머지는 같이 쓴다. forward의 `task` 인자가 어느 쪽을 쓸지 고른다. 분류 출력 수는
  `n_out`, 회귀 구간 수는 `n_buckets`
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from minipfn.models.cell_transformer import NAN_INDICATOR, standardize_cells

TASKS = ("classification", "regression", "multitask")
ENCODERS = ("linear", "fourier")
N_FOURIER = 32  # TabPFN-3.5 cell encoder의 주파수 수
N_HARMONICS = 4  # TabPFN-3.5 ECDF의 sin·cos 차수
# 추론에서 받는 클래스 수 상한 (공식 TabPFN-3.5와 같다). retrieval decoder에는 클래스마다
# 파라미터가 없어서, 정답 embedding 수(`n_out`)보다 많은 클래스는 embedding을 묶어 받는다
# (`forward`의 `y_classes`)
MAX_CLASSES = 160


@dataclass
class ColumnRowTransformerConfig:
    """모델 크기. 기본값은 TabPFN-3 기본 모델(Appendix C 표 2~7)."""

    # 출력: 분류 retrieval decoder, 회귀 bar distribution, 또는 둘 다(multitask)
    task: str = "classification"
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
    target_norm: bool = False  # TabPFN-3.5: True
    head_mlp: bool = False  # TabPFN-3.5: True
    # 분류(multitask면 그 분류 쪽): 정답 embedding 수(클래스 상한). 회귀만이면 구간 수
    n_out: int = 10
    n_buckets: int = 5000  # multitask의 회귀 구간 수
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

    @property
    def tasks(self) -> tuple[str, ...]:
        """이 모델이 내는 task."""
        return ("classification", "regression") if self.task == "multitask" else (self.task,)

    def output_size(self, task: str) -> int:
        """그 task의 출력 수 (분류: 클래스 상한, 회귀: 구간 수)."""
        if task == "classification" or self.task != "multitask":
            return self.n_out
        return self.n_buckets

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def tiny(cls, **changes: Any) -> ColumnRowTransformerConfig:
        """손 계산으로 따라갈 만한 작은 설정 (노트북·테스트용): E 16, inducing 4개, 열 1블록,
        CLS 2개, 행 2블록, 행 ICL 2층, head 2개씩, 클래스 상한 3. 바꿀 필드는 인자로 준다."""
        values = {
            "emsize": 16,
            "n_inducing": 4,
            "column_blocks": 1,
            "column_heads": 2,
            "n_cls": 2,
            "row_blocks": 2,
            "row_heads": 2,
            "nlayers": 2,
            "nhead": 2,
            "decoder_heads": 2,
            "decoder_head_dim": 8,
            "qassmax_hidden": 8,
            "n_out": 3,
        }
        return cls(**(values | changes))


# 모델의 맨 앞 모듈 이름 → 단계 (`ColumnRowTransformer.num_parameters_by_stage`)
STAGES: dict[str, str] = {
    "cell_encoder": "cell_encoder",
    "cell_target": "cell_encoder",
    "cell_target_norm": "cell_encoder",
    "column_blocks": "column_stage",
    "cls_tokens": "row_stage",
    "row_blocks": "row_stage",
    "row_norm": "row_stage",
    "icl_target": "row_icl",
    "icl_target_norm": "row_icl",
    "icl_layers": "row_icl",
    "icl_norm": "row_icl",
    "pre_head": "output",
    "heads": "output",
}
STAGE_ORDER = ("cell_encoder", "column_stage", "row_stage", "row_icl", "output")


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


class QASSMax(nn.Module):
    """query-aware scalable softmax: q ← q · base_mlp(log n) · (1 + tanh(query_mlp(q)))
    (모듈 docstring). base_mlp는 key 수에 따른 (head, head dim)마다의 배율, query_mlp는 query에
    따른 조절이고 처음에는 0이다."""

    def __init__(self, nhead: int, head_dim: int, hidden: int) -> None:
        super().__init__()
        self.nhead, self.head_dim = nhead, head_dim
        self.base_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, nhead * head_dim)
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(head_dim, hidden), nn.GELU(), nn.Linear(hidden, head_dim)
        )
        nn.init.zeros_(self.query_mlp[2].weight)
        nn.init.zeros_(self.query_mlp[2].bias)

    def forward(self, q: torch.Tensor, n_keys: int) -> torch.Tensor:
        # q: (N, H, L, d)
        if torch.compiler.is_compiling():
            # compile할 때는 텐서로: Python float로 만들면 n마다 다시 compile한다
            log_n = (torch.ones((1, 1), device=q.device) * n_keys).clamp(min=1).log().to(q.dtype)
        else:  # 예전과 같은 값 (float64 log를 q의 dtype으로)
            log_n = q.new_full((1, 1), math.log(max(n_keys, 1)))
        base = self.base_mlp(log_n).view(1, self.nhead, 1, self.head_dim)
        return q * base * (1 + torch.tanh(self.query_mlp(q)))


class Attention(nn.Module):
    """multi-head attention. q·k·v를 한 weight(`qkv`, (3D, D))로 둔다.

    - `context`를 주면 key·value는 거기서(cross-attention), 아니면 query와 같은 입력에서
    - `n_context`를 주면(self-attention) 앞의 n_context개만 key·value가 되고, 그 뒤 query(테스트
      행)는 첫 head의 key·value만 본다(multi-query)
    - `qassmax`: QASSMax (모듈 docstring), `qk_norm`: q·k에 RMSNorm, `rope_base`: q·k에 RoPE
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
            self.q_norm = nn.RMSNorm(dim // nhead)
            self.k_norm = nn.RMSNorm(dim // nhead)
        if zero_out:
            nn.init.zeros_(self.out.weight)
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
            # query가 앞쪽 일부(CLS)여도 위치는 key와 같은 0부터다
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
        self.attention_norm = nn.RMSNorm(dim)
        self.mlp = _mlp(dim, ff_factor)
        self.mlp_norm = nn.RMSNorm(dim)

    def forward(
        self, h: torch.Tensor, n_context: int | None = None, n_queries: int | None = None
    ) -> torch.Tensor:
        """`n_queries`를 주면 앞의 n_queries개 토큰만 query로 계산해 그 출력만 낸다(key·value는
        모든 토큰. 행 단계 마지막 블록의 CLS)."""
        normed = self.attention_norm(h)
        if n_queries is None:
            h = h + self.attention(normed, n_context=n_context)
        else:
            h = h[:, :n_queries] + self.attention(normed[:, :n_queries], context=normed)
        return h + self.mlp(self.mlp_norm(h))


class InducedBlock(nn.Module):
    """열 단계 블록 하나 (ISAB): inducing point가 학습 행을 요약하고(QASSMax), 모든 행이 그 요약을
    본다."""

    def __init__(
        self,
        dim: int,
        nhead: int,
        n_inducing: int,
        ff_factor: int,
        qassmax_hidden: int,
        **attention,
    ) -> None:
        super().__init__()
        self.inducing = nn.Parameter(nn.init.trunc_normal_(torch.empty(n_inducing, dim), std=0.02))
        self.summarize = Attention(
            dim, nhead, qassmax=True, qassmax_hidden=qassmax_hidden, **attention
        )
        self.summarize_mlp = _mlp(dim, ff_factor)
        self.broadcast = Attention(dim, nhead, **attention)
        self.broadcast_mlp = _mlp(dim, ff_factor)
        # 학습 파라미터가 있으므로 query·context·MLP 앞을 따로 둔다
        self.norms = nn.ModuleDict(
            {name: nn.RMSNorm(dim) for name in ("ind", "rows", "ind_mlp", "h", "hidden", "h_mlp")}
        )

    def forward(self, h: torch.Tensor, n_train: int) -> torch.Tensor:
        # h: (N, R, E), N = 데이터셋 × 열. 앞의 n_train행이 학습 행
        return self.spread(h, self.summary(h[:, :n_train]))

    def summary(self, h_train: torch.Tensor) -> torch.Tensor:
        """inducing point가 학습 행 (N, n_train, E)을 요약한 것 (N, n_inducing, E). 테스트 행은 보지
        않으므로, 추론에서 행을 나눠 계산할 때 열마다 한 번만 구해 둔다."""
        nm = self.norms
        ind = self.inducing.to(h_train.dtype).expand(h_train.shape[0], -1, -1)
        ind = ind + self.summarize(nm["ind"](ind), context=nm["rows"](h_train))
        return ind + self.summarize_mlp(nm["ind_mlp"](ind))

    def spread(self, h: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        """행마다 요약(`summary`)에 attention한다. 행끼리는 서로 보지 않는다."""
        nm = self.norms
        h = h + self.broadcast(nm["h"](h), context=nm["hidden"](ind))
        return h + self.broadcast_mlp(nm["h_mlp"](h))


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


class TargetEncoder(nn.Module):
    """학습 행 정답 → embedding. 분류는 클래스마다 직교 초기화한 벡터, 회귀는 Linear(y)."""

    def __init__(self, task: str, n_classes: int, dim: int) -> None:
        super().__init__()
        self.task, self.n_classes = task, n_classes
        if task == "classification":
            self.encode = nn.Embedding(n_classes, dim)
            nn.init.orthogonal_(self.encode.weight)
        else:
            self.encode = nn.Linear(1, dim)

    def forward(self, y_train: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # (B, N) → (B, N, dim)
        if self.task == "classification":
            return self.encode(y_train.long().clamp(0, self.n_classes - 1)).to(dtype)
        return self.encode(y_train.to(dtype)[..., None])


class PreHeadMLP(nn.Module):
    """출력층 앞 잔차 MLP (pre-norm, 마지막 층 0 시작이라 처음에는 항등)."""

    def __init__(self, dim: int, ff_factor: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.mlp = _mlp(dim, ff_factor)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.mlp(self.norm(h))


class _CellStats:
    """셀 정규화(`standardize_cells`)와 ECDF midrank(`ecdf_midrank`)의 학습 행 통계를 한 번 구해
    두고, 행·열 일부(`features`)에 같은 식을 쓴다 (`ColumnRowTransformer._rows_chunked`). 칸마다의
    계산이라 전체 표로 한 번에 계산한 값과 같다."""

    def __init__(self, x: torch.Tensor, n_train: int, ecdf: bool) -> None:
        train = x[:, :n_train]
        finite = torch.isfinite(train)
        t = torch.where(finite, train, torch.nan)
        self.mean = torch.nanmean(t, dim=1, keepdim=True)  # (B, 1, C)
        var = torch.nanmean(torch.square(t - self.mean), dim=1, keepdim=True)
        self.std = torch.sqrt(torch.nan_to_num(var, nan=0.0))
        self.used = (self.std > 1e-6).to(x.dtype)
        self.sorted = self.n_valid = None
        if ecdf:
            train = torch.where(finite, train.float(), torch.inf)
            self.sorted = train.transpose(1, 2).sort(-1).values.contiguous()  # (B, C, N)
            self.n_valid = finite.sum(1).clamp_min(1)  # (B, C)

    def features(
        self, x: torch.Tensor, cols: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """행 일부 x (B, Rj, Cj)의 (z값, 결측 표시, ECDF u). `cols`는 x의 열이 원래 표의 몇 번째
        열인지(None이면 모든 열). u는 ECDF를 쓰지 않으면 None이다."""

        def pick(t: torch.Tensor) -> torch.Tensor:
            return t if cols is None else t[..., cols]

        mean, std, used = pick(self.mean), pick(self.std), pick(self.used)
        missing = ~torch.isfinite(x)
        z = torch.where(missing, torch.nan, x)
        z = torch.nan_to_num((z - mean) / std.clamp_min(1e-6), nan=0.0).clamp(-100.0, 100.0)
        indicator = missing.to(z.dtype) * NAN_INDICATOR
        z = z * used
        if self.sorted is None:
            return z, indicator, None
        sorted_train = self.sorted if cols is None else self.sorted[:, cols]
        n_valid = self.n_valid if cols is None else self.n_valid[:, cols]
        values = torch.where(~missing, x.float(), 0.0).transpose(1, 2).contiguous()
        left = torch.searchsorted(sorted_train, values, side="left")
        right = torch.searchsorted(sorted_train, values, side="right")
        u = ((left + right).float() / 2 / n_valid[..., None].float()).clamp(0, 1)
        u = torch.where(~missing, u.transpose(1, 2), 0.5).to(z.dtype)
        return z, indicator, torch.where(used > 0, u, 0.5)


class RetrievalDecoder(nn.Module):
    """many-class retrieval decoder: 테스트 행(query)이 학습 행(key)에 attention한 가중치로 학습
    행의 one-hot 정답을 평균한다(head 평균). logit은 log(max(p, 1e-5) + 3e-5)."""

    def __init__(self, dim: int, heads: int, head_dim: int, n_classes: int, hidden: int) -> None:
        super().__init__()
        self.heads, self.head_dim, self.n_classes = heads, head_dim, n_classes
        self.q_projection = nn.Linear(dim, heads * head_dim)
        self.k_projection = nn.Linear(dim, heads * head_dim)
        self.qassmax = QASSMax(heads, head_dim, hidden)

    def forward(
        self,
        h: torch.Tensor,
        y_train: torch.Tensor,
        n_train: int,
        *,
        n_classes: int | None = None,
        fused: bool = False,
    ) -> torch.Tensor:
        """h: (B, R, D) → 테스트 행 logit (B, T, n_classes). query는 테스트 행, key는 학습 행에서만.

        `n_classes`(기본: 정답 embedding 수)는 one-hot의 클래스 수다. decoder에는 클래스마다
        파라미터가 없어서 정답 embedding 수보다 많아도 된다(추론에서 정답 embedding을 묶을 때).
        `fused`면 (테스트 행 × 학습 행) score 행렬을 만들지 않고 one-hot을 value로 둔
        attention 한 번으로 계산한다(추론용. 큰 표에서 그 행렬이 수백 GB가 된다). 값은 같고
        계산 순서만 다르다."""
        b = h.shape[0]
        n_classes = n_classes or self.n_classes
        q = self.q_projection(h[:, n_train:]).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_projection(h[:, :n_train]).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        q = self.qassmax(q, n_train)
        onehot = F.one_hot(y_train.long().clamp(0, n_classes - 1), n_classes).float()
        if fused:
            p = _retrieve(q, k, onehot).mean(1)
        else:
            # 확률·log는 float32로 (bf16이면 clip 값 근처가 뭉개진다)
            scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.head_dim)
            p = (torch.softmax(scores, dim=-1) @ onehot[:, None]).mean(1)  # (B, T, n_classes)
        return torch.log(p.clamp_min(1e-5) + 3e-5)


def _retrieve(q: torch.Tensor, k: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """softmax(q·kᵀ/√d) @ onehot을 head마다 (B, H, T, n_classes)로, score 행렬 없이 float32로.

    q (B, H, T, d), k (B, H, N, d), onehot (B, N, n_classes). attention kernel의 value 폭을 q·k와
    같게 하려고 one-hot을 d 열씩 잘라 batch 쪽으로 접는다(공식 TabPFN-3.5 추론 코드의
    `_chunked_class_attention`과 같은 방법). autocast를 끄고 float32로 계산한다."""
    b, heads, t, d = q.shape
    n, n_classes = onehot.shape[1:]
    pieces = math.ceil(n_classes / d)
    v = F.pad(onehot, (0, pieces * d - n_classes)).view(b, n, pieces, d).permute(0, 2, 1, 3)
    v = v[:, :, None].expand(b, pieces, heads, n, d).reshape(b * pieces, heads, n, d)
    q = q.float()[:, None].expand(b, pieces, heads, t, d).reshape(b * pieces, heads, t, d)
    k = k.float()[:, None].expand(b, pieces, heads, n, d).reshape(b * pieces, heads, n, d)
    with torch.autocast(q.device.type, enabled=False):
        out = F.scaled_dot_product_attention(q, k, v.contiguous())  # (B × pieces, H, T, d)
    out = out.view(b, pieces, heads, t, d).permute(0, 2, 3, 1, 4).reshape(b, heads, t, -1)
    return out[..., :n_classes]


class ColumnRowTransformer(nn.Module):
    # 모델 종류 (`minipfn.models.MODEL_TYPES`)
    model_type = "column_row_transformer"
    config_class = ColumnRowTransformerConfig
    # 추론(`forward`의 `inference`)에서 단계 0~2를 나눠 계산하는 크기. 공식 TabPFN-3.5 추론 코드의
    # 기본값과 같다: 행 묶음 하나의 셀 수(batch × 행 × 열), 학습 행 요약을 구할 때 한 번에 계산하는
    # 열 수
    inference_chunk_cells = 2048 * 768
    inference_column_chunk = 4
    # forward가 추론용 인자(`inference`, `y_classes`·`n_classes`)를 받는다
    # (`minipfn.inference.ensemble.forward_kwargs`)
    supports_inference_options = True

    def __init__(self, config: ColumnRowTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or ColumnRowTransformerConfig()
        e, g, d = config.emsize, config.features_per_group, config.icl_dim
        self.offsets = group_offsets(g)
        if config.encoder == "fourier":
            self.cell_encoder = FourierCellEncoder(g, e)
        else:
            self.cell_encoder = nn.Linear(2 * g, e)  # [값 g개, 결측 표시 g개] → 토큰
        # 정답 encoder: 셀 단계·행 ICL 입력 각각, task마다
        self.cell_target = nn.ModuleDict(
            {t: TargetEncoder(t, config.n_out, e) for t in config.tasks}
        )
        self.icl_target = nn.ModuleDict(
            {t: TargetEncoder(t, config.n_out, d) for t in config.tasks}
        )
        if config.target_norm:  # 두 task가 같이 쓴다
            self.cell_target_norm = nn.LayerNorm(e)
            self.icl_target_norm = nn.LayerNorm(d)
        self.column_blocks = nn.ModuleList(
            InducedBlock(
                e,
                config.column_heads,
                config.n_inducing,
                config.ff_factor,
                config.qassmax_hidden,
                qk_norm=config.qk_norm,
            )
            for _ in range(config.column_blocks)
        )
        self.cls_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(config.n_cls, e), std=0.02)
        )
        self.row_blocks = nn.ModuleList(
            Block(
                e,
                config.row_heads,
                config.ff_factor,
                qk_norm=config.qk_norm,
                rope_base=config.rope_base,
                zero_out=False,  # 모듈 docstring "여기서 정한 것"
            )
            for _ in range(config.row_blocks)
        )
        self.row_norm = nn.RMSNorm(e)
        qassmax = {"qassmax": True, "qassmax_hidden": config.qassmax_hidden}
        self.icl_layers = nn.ModuleList(
            Block(d, config.nhead, config.ff_factor, qk_norm=config.qk_norm, **qassmax)
            for _ in range(config.nlayers)
        )
        self.icl_norm = nn.RMSNorm(d)
        if config.head_mlp:
            self.pre_head = nn.ModuleDict(
                {t: PreHeadMLP(d, config.ff_factor) for t in config.tasks}
            )
        self.heads = nn.ModuleDict()
        for t in config.tasks:
            n = config.output_size(t)
            if t == "classification":
                self.heads[t] = RetrievalDecoder(
                    d, config.decoder_heads, config.decoder_head_dim, n, config.qassmax_hidden
                )
            elif config.head_mlp:
                self.heads[t] = nn.Linear(d, n)
            else:  # TabPFN-3 회귀: D → 2D → GELU → 구간 수
                self.heads[t] = nn.Sequential(
                    nn.Linear(d, config.ff_factor * d),
                    nn.GELU(),
                    nn.Linear(config.ff_factor * d, n),
                )

    @property
    def tasks(self) -> tuple[str, ...]:
        """이 모델이 내는 task (multitask면 둘 다. forward의 `task`로 고른다)."""
        return self.config.tasks

    @property
    def max_classes(self) -> int:
        """추론에서 받는 클래스 수 상한 (`MAX_CLASSES`, 정답 embedding 수가 더 많으면 그 수)."""
        return max(self.config.n_out, MAX_CLASSES)

    def repeated_blocks(self) -> list[nn.Module]:
        """여러 번 쓰는 블록 (학습 가속 옵션 `compile`이 블록마다 compile한다)."""
        return [*self.column_blocks, *self.row_blocks, *self.icl_layers]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_parameters_by_stage(self) -> dict[str, int]:
        """단계마다 파라미터 수 (`STAGE_ORDER` 순서): 셀 encoder(정답 embedding 포함), 열 단계,
        행 단계, 행 ICL(정답 embedding 포함), 출력."""
        counts = dict.fromkeys(STAGE_ORDER, 0)
        for name, p in self.named_parameters():
            counts[STAGES[name.split(".")[0]]] += p.numel()
        return counts

    def num_parameters_by_task(self) -> dict[str, int]:
        """두 task가 같이 쓰는 파라미터("shared")와 task마다 따로인 파라미터 수. 이름에 task가
        들어간 부품(정답 encoder, 출력층 앞 MLP, 출력층)이 그 task 것이다."""
        counts = {"shared": 0} | dict.fromkeys(self.config.tasks, 0)
        for name, p in self.named_parameters():
            parts = name.split(".")
            task = next((t for t in self.config.tasks if t in parts), "shared")
            counts[task] += p.numel()
        return counts

    def qk_clip_weights(self) -> list[tuple[str, torch.Tensor]]:
        """빈 목록: QK-clip의 가중치 상한이 이 모델의 logit을 묶지 못한다 (QASSMax의 입력에 따른
        배율, 테스트 행의 multi-query. 모듈 docstring). 상한을 지표로 남기면 실제보다 안전해
        보이므로 지표도 내지 않는다."""
        return []

    def decoder_weight(self) -> torch.Tensor:
        """학습 지표 `decoder_norm`: 회귀 출력층(있으면), 아니면 분류 query 가중치."""
        if "regression" in self.heads:
            head = self.heads["regression"]
            return head.weight if isinstance(head, nn.Linear) else head[-1].weight
        return self.heads["classification"].q_projection.weight

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

    def _target(self, stage: str, task: str, y_train: torch.Tensor, rows: int, like: torch.Tensor):
        # 학습 행 정답 embedding (B, R, dim). 테스트 행은 0
        target = getattr(self, f"{stage}_target")[task](y_train, like.dtype)
        if self.config.target_norm:
            target = getattr(self, f"{stage}_target_norm")(target)
        return F.pad(target, (0, 0, 0, rows - y_train.shape[1]))

    def forward(
        self,
        x: torch.Tensor,
        y_train: torch.Tensor,
        task: str | None = None,
        *,
        y_classes: torch.Tensor | None = None,
        n_classes: int | None = None,
        inference: bool = False,
    ) -> torch.Tensor:
        """테스트 행의 logit (B, R - n_train, 출력 수).

        x: (B, R, C), 학습 행 다음에 테스트 행. y_train: (B, n_train).
        분류면 y는 0부터 시작하는 클래스 번호, 회귀면 z-정규화한 값이다. `task`는 multitask
        모델에서만 준다("classification" 또는 "regression"). 한 task 모델은 그 task다.

        추론에서만 쓰는 인자:

        - `y_classes`·`n_classes`(분류): retrieval decoder가 평균할 학습 행 클래스 번호와 클래스 수.
          주면 `y_train`은 정답 embedding에만 쓰인다. 클래스가 정답 embedding 수(`n_out`)보다
          많을 때 embedding 번호를 묶어 `y_train`으로, 원래 번호를 `y_classes`로 준다
          (`minipfn.inference.ensemble.EnsembleClassifier`). 출력은 (B, T, n_classes)
        - `inference`: 큰 표의 메모리를 줄인다. 셀 수(B × R × C)가 `inference_chunk_cells`보다
          많으면 단계 0~2를 행 묶음으로 나눠 계산하고(`_rows_chunked`), 분류 decoder는 score 행렬을
          만들지 않는다. 테스트 행끼리는 서로 보지 않으므로 값은 같다(계산 순서만 다르다)
        """
        cfg = self.config
        if task is None:
            if cfg.task == "multitask":
                raise ValueError("multitask 모델은 task를 준다 (classification 또는 regression)")
            task = cfg.task
        elif task not in cfg.tasks:
            raise ValueError(f"이 모델의 task는 {cfg.tasks}다: {task}")
        if n_classes is not None and n_classes > self.max_classes:
            raise ValueError(f"클래스 {n_classes}개는 상한({self.max_classes}개)보다 많다")
        b, r, c = x.shape
        n_train = y_train.shape[1]
        if inference and b * r * c > self.inference_chunk_cells:
            rows = self._rows_chunked(x, y_train, task)
        else:
            rows = self._rows(x, y_train, task)

        # 3) 행 ICL: 학습 행 벡터에 정답 embedding을 한 번 더 더한다
        rows = rows + self._target("icl", task, y_train, r, rows)
        for layer in self.icl_layers:
            rows = layer(rows, n_context=n_train)
        rows = self.icl_norm(rows)

        # 4) 출력
        if cfg.head_mlp:
            rows = self.pre_head[task](rows)
        if task == "classification":
            labels = y_train if y_classes is None else y_classes
            return self.heads[task](rows, labels, n_train, n_classes=n_classes, fused=inference)
        return self.heads[task](rows[:, n_train:])

    def _rows(self, x: torch.Tensor, y_train: torch.Tensor, task: str) -> torch.Tensor:
        """단계 0~2 (셀 encoder, 열 단계, 행 단계): x (B, R, C) → 행 벡터 (B, R, n_cls × E)."""
        b, r, c = x.shape
        n_train = y_train.shape[1]
        cells = self._embed_cells(x, n_train)  # (B, R, C, E)
        cells = cells + self._target("cell", task, y_train, r, cells)[:, :, None]
        e = cells.shape[-1]

        # 1) 열 단계: (B × C, R, E)
        h = cells.permute(0, 2, 1, 3).reshape(b * c, r, e)
        for block in self.column_blocks:
            h = block(h, n_train)
        # 2) 행 단계
        return self._row_stage(h.view(b, c, r, e).transpose(1, 2))

    def _row_stage(self, h: torch.Tensor) -> torch.Tensor:
        """행 단계: 셀 토큰 (B, R, C, E) → (B × R, n_cls + C, E) → CLS 출력을 이어 붙인 행 벡터
        (B, R, n_cls × E). 마지막 블록은 CLS만 query다(그 뒤에 쓰는 것은 CLS뿐)."""
        cfg = self.config
        b, r, c, e = h.shape
        h = h.reshape(b * r, c, e)
        h = torch.cat([self.cls_tokens.to(h.dtype).expand(b * r, -1, -1), h], dim=1)
        for block in self.row_blocks[:-1]:
            h = block(h)
        h = self.row_blocks[-1](h, n_queries=cfg.n_cls)
        return self.row_norm(h).reshape(b, r, cfg.icl_dim)

    def _rows_chunked(self, x: torch.Tensor, y_train: torch.Tensor, task: str) -> torch.Tensor:
        """`_rows`와 같은 값을 행 묶음으로 나눠 계산한다 (추론용, 공식 TabPFN-3.5 추론 코드의
        chunkwise inference와 같은 순서).

        열 단계에서 행은 서로 보지 않고 inducing point의 요약(`InducedBlock.summary`)만 본다. 요약은
        학습 행으로만 정해지므로 (1) 열을 `inference_column_chunk`개씩 나눠 학습 행만으로 블록마다
        요약을 구해 두고, (2) 행을 `inference_chunk_cells / (B × C)`개씩 나눠 셀 encoder, 열
        단계(요약에 attention), 행 단계를 계산한다. 셀 정규화·ECDF 통계도 학습 행으로 한 번만
        구한다(`_CellStats`). 그래서 (B, R, C, E) 크기의 텐서를 한 번에 두지 않는다."""
        b, r, c = x.shape
        n_train = y_train.shape[1]
        stats = _CellStats(x, n_train, ecdf=self.config.encoder == "fourier")
        offsets = torch.tensor(self.offsets, device=x.device)
        groups = (torch.arange(c, device=x.device)[:, None] + offsets) % c  # 열 j에 묶는 열
        target = None  # 학습 행의 셀 정답 embedding (B, n_train, E), 셀 dtype을 알 때 만든다

        def embed(x_part: torch.Tensor, cols: torch.Tensor | None, shape) -> torch.Tensor:
            nonlocal target
            cells = self._encode_cells(*stats.features(x_part, cols), shape)
            if target is None:
                target = self._target("cell", task, y_train, n_train, cells)
            return cells

        # (1) 블록마다 요약 (B × C, n_inducing, E)
        summaries: list[torch.Tensor | None] = [None] * len(self.column_blocks)
        for c0 in range(0, c, self.inference_column_chunk):
            src = groups[c0 : c0 + self.inference_column_chunk]  # (Cj, G): 묶는 열
            cj = src.shape[0]
            cells = embed(x[:, :n_train, src.reshape(-1)], src.reshape(-1), (cj, src.shape[1]))
            cells = cells + target[:, :, None]
            e = cells.shape[-1]
            h = cells.permute(0, 2, 1, 3).reshape(b * cj, n_train, e)
            for i, block in enumerate(self.column_blocks):
                ind = block.summary(h)
                if summaries[i] is None:
                    summaries[i] = ind.new_empty(b, c, *ind.shape[1:])
                summaries[i][:, c0 : c0 + cj] = ind.view(b, cj, *ind.shape[1:])
                if i + 1 < len(self.column_blocks):
                    h = block.spread(h, ind)

        # (2) 행 묶음마다 셀 encoder → 열 단계 → 행 단계
        step = max(1, self.inference_chunk_cells // (b * c))
        out = None
        for r0 in range(0, r, step):
            r1 = min(r0 + step, r)
            cells = embed(x[:, r0:r1], None, None)  # (B, Rj, C, E)
            if r0 < n_train:  # 학습 행에만 정답 embedding
                k = min(r1, n_train) - r0
                cells = torch.cat([cells[:, :k] + target[:, r0 : r0 + k, None], cells[:, k:]], 1)
            e = cells.shape[-1]
            h = cells.permute(0, 2, 1, 3).reshape(b * c, r1 - r0, e)
            for block, ind in zip(self.column_blocks, summaries, strict=True):
                h = block.spread(h, ind.reshape(b * c, *ind.shape[2:]))
            rows = self._row_stage(h.view(b, c, r1 - r0, e).transpose(1, 2))
            if out is None:
                out = rows.new_empty(b, r, rows.shape[-1])
            out[:, r0:r1] = rows
        return out

    def _encode_cells(self, z, indicator, u, shape) -> torch.Tensor:
        """셀 값 (`_CellStats.features`) → 토큰. `shape`이 None이면 열 C개를 circular shift로
        묶고(`_group`), (Cj, G)면 이미 묶은 열 Cj × G개다. 결과는 (B, 행, Cj, E)."""

        def group(t: torch.Tensor) -> torch.Tensor:
            if shape is None:
                return _group(t, self.offsets)
            return t.view(*t.shape[:2], *shape)

        zg, ig = group(z), group(indicator)
        if self.config.encoder == "fourier":
            return self.cell_encoder(zg, ig, group(u))
        return self.cell_encoder(torch.cat([zg, ig], dim=-1))
