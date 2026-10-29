"""TabPFN v2 Transformer (Nature 2025, Fig. 1b와 Methods "Details on the neural architecture").

논문 설명대로 만든 구조다. 각 부분은 `notebooks/03_tabpfn_v2/01_two_way_attention_primer`에서
작은 예로 따라갔다.

- 입력: feature마다 학습 행의 평균·표준편차로 z-정규화하고, 결측 칸은 0으로 두고 표시를 따로 넣는다
- 셀 2개를 묶어 토큰 하나로 만들고(Linear), 마지막 열에 y 토큰을 붙인다: (B, R, G + 1, E)
- 열 이름표: 열마다 고정 시드 무작위 벡터(E/4) → 학습하는 Linear → 그 열의 모든 토큰에 더한다
- 층마다 feature attention(같은 행) → sample attention(같은 열) → MLP. 각각 잔차를 더한 뒤 LayerNorm
- sample attention: 학습 행은 학습 행끼리 attend하고, 테스트 행은 학습 행에만 attend한다.
  테스트 행은 key·value의 첫 head만 쓴다(논문의 multi-query 변형)
- 출력: 테스트 행의 y 토큰 → MLP → 분류는 클래스 logit 10개, 회귀는 bar distribution 구간 logit
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

NAN_INDICATOR = -2.0  # 결측 칸의 표시 값 (아닌 칸은 0)


@dataclass
class TabPFNv2Config:
    """모델 크기. 기본값은 논문 기본 모델의 크기(emsize 192, 12층, head 6, 셀 2개 묶기)."""

    emsize: int = 192
    nlayers: int = 12
    nhead: int = 6
    nhid_factor: int = 4
    features_per_group: int = 2
    n_out: int = 10  # 분류: 최대 클래스 수, 회귀: bar distribution 구간 수
    column_embedding_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, emsize: int, nhead: int) -> None:
        super().__init__()
        self.nhead = nhead
        self.qkv = nn.Linear(emsize, 3 * emsize, bias=False)
        self.out = nn.Linear(emsize, emsize, bias=False)
        nn.init.zeros_(self.out.weight)  # 처음에는 attention이 잔차에 아무것도 더하지 않는다

    def forward(self, x: torch.Tensor, n_context: int | None = None) -> torch.Tensor:
        """x: (N, L, E). n_context를 주면 앞의 n_context개(학습 행)만 key·value가 된다."""
        n, length, e = x.shape
        q, k, v = self.qkv(x).view(n, length, 3, self.nhead, e // self.nhead).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (N, H, L, D)
        if n_context is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            k, v = k[:, :, :n_context], v[:, :, :n_context]
            train = F.scaled_dot_product_attention(q[:, :, :n_context], k, v)
            # 테스트 행: 모든 query head가 첫 head의 key·value를 본다
            first_k, first_v = k[:, :1].expand_as(k), v[:, :1].expand_as(v)
            test = F.scaled_dot_product_attention(q[:, :, n_context:], first_k, first_v)
            out = torch.cat([train, test], dim=2)
        return self.out(out.transpose(1, 2).reshape(n, length, e))


class TabPFNv2Layer(nn.Module):
    def __init__(self, emsize: int, nhead: int, nhid: int) -> None:
        super().__init__()
        self.feature_attention = Attention(emsize, nhead)
        self.sample_attention = Attention(emsize, nhead)
        self.mlp = nn.Sequential(
            nn.Linear(emsize, nhid, bias=False), nn.GELU(), nn.Linear(nhid, emsize, bias=False)
        )
        nn.init.zeros_(self.mlp[2].weight)
        self.norms = nn.ModuleList(nn.LayerNorm(emsize, elementwise_affine=False) for _ in range(3))

    def forward(self, h: torch.Tensor, n_train: int) -> torch.Tensor:
        # h: (B, R, C, E), C = feature 묶음 수 + 1 (y 토큰)
        b, r, c, e = h.shape

        # 1) feature attention: 행 하나의 토큰 C개끼리
        out = self.feature_attention(h.reshape(b * r, c, e)).reshape(b, r, c, e)
        h = self.norms[0](h + out)

        # 2) sample attention: 열 하나의 행 R개끼리 (테스트 행은 학습 행에만)
        cols = h.transpose(1, 2).reshape(b * c, r, e)
        out = self.sample_attention(cols, n_context=n_train).reshape(b, c, r, e).transpose(1, 2)
        h = self.norms[1](h + out)

        # 3) MLP: 토큰마다
        return self.norms[2](h + self.mlp(h))


class TabPFNv2(nn.Module):
    def __init__(self, config: TabPFNv2Config | None = None) -> None:
        super().__init__()
        self.config = config = config or TabPFNv2Config()
        e, g = config.emsize, config.features_per_group
        self.feature_embedder = nn.Linear(2 * g, e)  # [값 g개, 결측 표시 g개] → 토큰
        self.target_embedder = nn.Linear(2, e)  # [y, 결측 표시] → 토큰
        self.column_embedder = nn.Linear(e // 4, e)
        nhid = e * config.nhid_factor
        self.layers = nn.ModuleList(
            TabPFNv2Layer(e, config.nhead, nhid) for _ in range(config.nlayers)
        )
        self.decoder = nn.Sequential(nn.Linear(e, nhid), nn.GELU(), nn.Linear(nhid, config.n_out))
        self._column_noise: dict[tuple, torch.Tensor] = {}  # state_dict에 넣지 않는 캐시

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _embed_features(self, x: torch.Tensor, n_train: int) -> torch.Tensor:
        # x: (B, R, F) → 토큰 (B, R, G, E)
        b, r, f = x.shape
        g = self.config.features_per_group

        # 1) 결측 표시를 만들고, 학습 행으로 z-정규화한다 (결측 칸은 평균, 즉 0이 된다)
        missing = ~torch.isfinite(x)
        x = torch.where(missing, torch.nan, x)
        mean = torch.nanmean(x[:, :n_train], dim=1, keepdim=True)
        var = torch.nanmean(torch.square(x[:, :n_train] - mean), dim=1, keepdim=True)
        std = torch.sqrt(torch.nan_to_num(var, nan=0.0))
        x = torch.nan_to_num((x - mean) / std.clamp_min(1e-6), nan=0.0)
        # 학습 행에서 거의 상수인 열은 다른 행에서 값이 아주 커질 수 있다. bf16 역전파가 넘치지
        # 않도록 ±100으로 자른다
        x = x.clamp(-100.0, 100.0)
        indicator = missing.to(x.dtype) * NAN_INDICATOR

        # 2) 학습 행에서 상수인 열은 정보가 없으므로 0으로 둔다
        used = (std > 1e-6).to(x.dtype)
        x = x * used

        # 3) 열 수를 묶음 크기의 배수로 맞추고 묶는다. 쓴 열이 적은 묶음은 √(g / 쓴 열 수)배
        pad = -f % g
        if pad:
            x, indicator, used = (
                F.pad(x, (0, pad)),
                F.pad(indicator, (0, pad)),
                F.pad(used, (0, pad)),
            )
        groups = (f + pad) // g
        x = x.view(b, r, groups, g)
        indicator = indicator.view(b, r, groups, g)
        n_used = used.view(b, 1, groups, g).sum(-1, keepdim=True).clamp_min(1)
        x = x * torch.sqrt(g / n_used)

        # 4) [값, 결측 표시] → 토큰
        return self.feature_embedder(torch.cat([x, indicator], dim=-1))

    def _embed_target(self, y_train: torch.Tensor, n_rows: int) -> torch.Tensor:
        # y_train: (B, n_train) → 토큰 (B, R, E). 테스트 행의 y는 빈 칸
        n_train = y_train.shape[1]
        y = F.pad(y_train, (0, n_rows - n_train))
        indicator = torch.zeros_like(y)
        indicator[:, n_train:] = NAN_INDICATOR
        return self.target_embedder(torch.stack([y, indicator], dim=-1))

    def column_embeddings(self, n_groups: int, like: torch.Tensor) -> torch.Tensor:
        # 열마다 고정된 무작위 벡터. CPU에서 뽑아 장치와 상관없이 같은 값을 쓴다.
        # 묶음 수마다 한 번만 뽑아 입력과 같은 장치·dtype으로 캐시해 둔다
        key = (n_groups, like.device, like.dtype)
        if key not in self._column_noise:
            generator = torch.Generator().manual_seed(self.config.column_embedding_seed)
            noise = torch.randn(n_groups, self.config.emsize // 4, generator=generator)
            self._column_noise[key] = noise.to(device=like.device, dtype=like.dtype)
        return self.column_embedder(self._column_noise[key])

    def forward(self, x: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        """테스트 행의 logit (B, R - n_train, n_out).

        x: (B, R, F), 학습 행 다음에 테스트 행. y_train: (B, n_train).
        분류면 y는 0부터 시작하는 클래스 번호, 회귀면 z-정규화한 값이다.
        """
        n_train = y_train.shape[1]
        x_tokens = self._embed_features(x, n_train)
        x_tokens = x_tokens + self.column_embeddings(x_tokens.shape[2], x_tokens)
        y_tokens = self._embed_target(y_train.to(x_tokens.dtype), x.shape[1])
        h = torch.cat([x_tokens, y_tokens.unsqueeze(2)], dim=2)  # (B, R, G + 1, E)
        for layer in self.layers:
            h = layer(h, n_train)
        return self.decoder(h[:, n_train:, -1])


def save_checkpoint(path: str | Path, model: TabPFNv2, borders: torch.Tensor | None = None) -> None:
    """가중치, 설정, (회귀면) bar distribution 경계를 저장한다."""
    payload = {
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "borders": borders,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[TabPFNv2, torch.Tensor | None]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = TabPFNv2(TabPFNv2Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["borders"]
