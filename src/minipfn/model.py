"""PFN 모델 — set을 처리하는 Transformer *encoder* (노트북 03 §7).

각 데이터 점 (x, y)가 토큰 하나. context 토큰은 실제 y를, query 토큰은 y 대신 학습형 mask
토큰을 쓴다(정답을 미리 못 봄). positional encoding이 없어 순서에 불변(permutation invariant)
이고, attention mask로 **context는 서로 full attention, query는 context에만 attend**하게 한다
(query끼리·자기 타깃은 안 봄). 이렇게 하면 한 번에 여러 query를 넣어도 서로 독립이며, 학습된
모델이 임의 개수의 context/query에 그대로 대응한다. 출력 head는 query마다 bar distribution
logits를 낸다. `automl/PFNs`의 `single_eval_pos`(= context 길이) 규약을 따른다.
"""

from __future__ import annotations

import torch
from torch import nn


def build_attention_mask(
    n_context: int, n_query: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """(S, S) bool mask. True = attend 금지.

    모든 토큰은 context(앞 n_context개)에만 attend한다. query 토큰은 자기 자신에도 attend해
    (y 정보가 없어 누출 없음) context가 0개일 때도 attention이 정의되게 한다 — query 독립성과
    순열 불변은 유지된다.
    """
    total = n_context + n_query
    idx = torch.arange(total, device=device)
    attend_context = idx[None, :] < n_context  # j < n_context
    attend_self = idx[:, None] == idx[None, :]
    allowed = attend_context | attend_self
    return ~allowed


class PFNTransformer(nn.Module):
    """1D 회귀용 PFN. forward(x_context, y_context, x_query) -> query별 bar-dist logits."""

    def __init__(
        self,
        x_dim: int = 1,
        emsize: int = 32,
        nhead: int = 2,
        nhid: int = 128,
        nlayers: int = 2,
        num_buckets: int = 100,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.emsize = emsize
        self.num_buckets = num_buckets
        self.x_encoder = nn.Linear(x_dim, emsize)
        self.y_encoder = nn.Linear(1, emsize)
        self.query_mask_token = nn.Parameter(torch.zeros(emsize))
        layer = nn.TransformerEncoderLayer(
            d_model=emsize,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False)
        self.head = nn.Linear(emsize, num_buckets)

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
    ) -> torch.Tensor:
        """(B, nc, x_dim), (B, nc), (B, nq, x_dim) -> logits (B, nq, num_buckets)."""
        n_context = x_context.shape[1]
        n_query = x_query.shape[1]

        ctx_tokens = self.x_encoder(x_context) + self.y_encoder(y_context.unsqueeze(-1))
        qry_tokens = self.x_encoder(x_query) + self.query_mask_token
        tokens = torch.cat([ctx_tokens, qry_tokens], dim=1)  # (B, S, emsize)

        mask = build_attention_mask(n_context, n_query, device=tokens.device)
        out = self.encoder(tokens, mask=mask)
        query_out = out[:, n_context:, :]  # single_eval_pos 이후가 query
        return self.head(query_out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
