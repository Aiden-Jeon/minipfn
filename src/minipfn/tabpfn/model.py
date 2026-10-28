"""TabPFN Transformer (논문 3절·부록 E.1, v1 `transformer.py`·`layer.py` 구조).

GP 재현용 `minipfn.model.PFNTransformer`와 다른 점은 v1을 따른 것이다.
- post-norm 층(잔차 더한 뒤 LayerNorm), GELU, attention 출력과 FFN 두 번째 Linear를 0으로 초기화
- 테스트 행 토큰은 x 임베딩만 쓴다(마스크 토큰 없음)
- 출력은 Linear → GELU → Linear decoder, logit 10개(최대 클래스 수)
- attention은 학습 행끼리, 테스트 행 → 학습 행만 계산한다(v1 efficient_eval_masking, 논문 n² + n·m).
  두 번 나눠 부르므로 mask 없이 PyTorch의 빠른 attention 커널을 쓸 수 있다.
- 추론용 KV 캐시: `build_kv_cache`로 층별 학습 행 key·value를 한 번 계산해 두면
  `forward_cached`는 테스트 행 → 학습 행 attention만 계산한다(n·m).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from minipfn.tabpfn import spec


class TabPFNLayer(nn.Module):
    def __init__(self, emsize: int, nhead: int, nhid: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(emsize, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(emsize, nhid)
        self.linear2 = nn.Linear(nhid, emsize)
        self.norm1 = nn.LayerNorm(emsize)
        self.norm2 = nn.LayerNorm(emsize)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, h: torch.Tensor, n_train: int) -> torch.Tensor:
        train, test = h[:, :n_train], h[:, n_train:]
        if n_train == 0:
            return torch.zeros_like(h)  # 참고할 학습 행이 없다
        train_out = self.self_attn(train, train, train, need_weights=False)[0]
        test_out = self.self_attn(test, train, train, need_weights=False)[0]
        return torch.cat([train_out, test_out], dim=1)

    def _finish(self, h: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        h = self.norm1(h + self.dropout(attn))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(h))))
        return self.norm2(h + self.dropout(ff))

    def forward(self, h: torch.Tensor, n_train: int) -> torch.Tensor:
        return self._finish(h, self._attend(h, n_train))

    # KV 캐시 경로: self_attn과 같은 가중치로 q·k·v를 직접 계산한다.
    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, e = x.shape
        nhead = self.self_attn.num_heads
        return x.view(b, n, nhead, e // nhead).transpose(1, 2)  # (B, H, n, d)

    def project_kv(self, train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """학습 행 (B, n, E)의 key·value (B, H, n, d)."""
        _, k, v = self.self_attn.in_proj_weight.chunk(3)
        _, bk, bv = self.self_attn.in_proj_bias.chunk(3)
        return self._heads(F.linear(train, k, bk)), self._heads(F.linear(train, v, bv))

    def forward_cached(self, h: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """h의 행들이 key·value (k, v)의 학습 행만 참고하는 한 층. h: (B, m, E)."""
        if k.shape[2] == 0:
            return self._finish(h, torch.zeros_like(h))
        w, b = self.self_attn.in_proj_weight.chunk(3)[0], self.self_attn.in_proj_bias.chunk(3)[0]
        out = F.scaled_dot_product_attention(self._heads(F.linear(h, w, b)), k, v)
        out = out.transpose(1, 2).reshape(h.shape)
        return self._finish(h, self.self_attn.out_proj(out))


class TabPFNTransformer(nn.Module):
    """forward(x_train, y_train, x_test) -> 테스트 행별 클래스 logits (B, n_test, n_out)."""

    def __init__(
        self,
        num_features: int = spec.MAX_FEATURES,
        n_out: int = spec.MAX_CLASSES,
        emsize: int = spec.EMSIZE,
        nhead: int = spec.NHEAD,
        nhid: int = spec.NHID,
        nlayers: int = spec.NLAYERS,
        dropout: float = spec.DROPOUT,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.n_out = n_out
        self.x_encoder = nn.Linear(num_features, emsize)
        self.y_encoder = nn.Linear(1, emsize)  # 클래스 번호를 실수 하나로 넣는다 (v1)
        self.layers = nn.ModuleList(
            TabPFNLayer(emsize, nhead, nhid, dropout) for _ in range(nlayers)
        )
        self.decoder = nn.Sequential(nn.Linear(emsize, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        for layer in self.layers:  # v1 init_weights
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

    def forward(
        self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor
    ) -> torch.Tensor:
        """(B, n, F), (B, n), (B, m, F) -> (B, m, n_out)."""
        n_train = x_train.shape[1]
        x = torch.nan_to_num(torch.cat([x_train, x_test], dim=1), nan=0.0)
        h = self.x_encoder(x)
        y_emb = self.y_encoder(y_train.unsqueeze(-1).to(h.dtype))
        h = torch.cat([h[:, :n_train] + y_emb, h[:, n_train:]], dim=1)
        for layer in self.layers:
            h = layer(h, n_train)
        return self.decoder(h[:, n_train:])

    # 학습 행은 학습 행만 참고하므로, 층마다 학습 행의 key·value는 테스트 행과 상관없이 정해진다.
    # 한 번 계산해 두면 테스트 행은 그것만 참고해 forward와 같은 logits를 낸다(추론 전용).
    def build_kv_cache(
        self, x_train: torch.Tensor, y_train: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """(B, n, F), (B, n) -> 층마다 학습 행 (key, value), 각각 (B, nhead, n, emsize/nhead)."""
        h = self.x_encoder(torch.nan_to_num(x_train, nan=0.0))
        h = h + self.y_encoder(y_train.unsqueeze(-1).to(h.dtype))
        cache = []
        for i, layer in enumerate(self.layers):
            k, v = layer.project_kv(h)
            cache.append((k, v))
            if i < len(self.layers) - 1:  # 마지막 층의 학습 행 출력은 쓰지 않는다
                h = layer.forward_cached(h, k, v)
        return cache

    def forward_cached(
        self, kv_cache: list[tuple[torch.Tensor, torch.Tensor]], x_test: torch.Tensor
    ) -> torch.Tensor:
        """build_kv_cache 결과와 (B, m, F) -> (B, m, n_out). forward와 같은 값."""
        h = self.x_encoder(torch.nan_to_num(x_test, nan=0.0))
        for layer, (k, v) in zip(self.layers, kv_cache, strict=True):
            h = layer.forward_cached(h, k, v)
        return self.decoder(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def configure_attention_backends() -> None:
    """GPU에서 cuDNN attention을 끈다.

    학습 행 수(single_eval_pos)가 forward마다 바뀌는데, cuDNN SDPA는 shape가 바뀔 때마다 실행
    그래프를 새로 만들어 호출당 수 ms의 CPU 시간이 든다(8×H100 프로파일에서 step 시간의 대부분).
    flash / memory-efficient 커널은 이런 비용이 없다.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)


def classification_loss(logits: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
    """테스트 행 cross-entropy. 무시할 행(-100)은 0으로 세고 전체 평균을 낸다 (v1과 같은 가중)."""
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        y_test.reshape(-1).long(),
        ignore_index=spec.IGNORE_INDEX,
        reduction="none",
    )
    return losses.mean()
