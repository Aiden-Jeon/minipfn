"""학습 run들이 같은 목표 검증 손실에 닿는 순수 학습 시간을 견준다 (issue #46).

목표는 기준 run(첫 번째)의 `--target-step` 검증 손실이다. 나머지 run이 그 값 이하가 된 첫 검증의
`train_seconds`(검증·checkpoint·기록을 뺀 학습 시간)를 기준과 비교한다. 같은 `val_every`로 학습한
run끼리 견준다(`minipfn.training.speed`). LAWA를 켠 run은 평균 가중치의 손실(`val_loss_avg`)로도
잰다.

예) 저장소 루트에서 (.env.local을 불러온 셸)
  MLFLOW_TRACKING_URI=databricks uv run python scripts/tabpfn_v3/time_to_target.py \
    --runs <기준 run_id> <run_id> ... --target-step 5000
"""

from __future__ import annotations

import argparse

from minipfn.training.speed import ValPoint, time_to_target, val_loss_at


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="MLflow run id, 첫 번째가 기준")
    parser.add_argument("--target-step", type=int, required=True, help="기준 run의 이 step 손실")
    parser.add_argument("--metric", default="val_loss", help="검증 손실 지표 (기본: val_loss)")
    return parser.parse_args(argv)


def run_points(client, run_id: str, metric: str) -> list[ValPoint]:
    """run의 검증 기록: 같은 step에 남은 `train_seconds`와 `metric`."""
    seconds = {m.step: m.value for m in client.get_metric_history(run_id, "train_seconds")}
    losses = {m.step: m.value for m in client.get_metric_history(run_id, metric)}
    return [ValPoint(s, seconds[s], losses[s]) for s in sorted(losses) if s in seconds]


def main(argv: list[str] | None = None) -> None:
    import mlflow

    args = parse_args(argv)
    client = mlflow.MlflowClient()
    base_points = run_points(client, args.runs[0], args.metric)
    target = val_loss_at(base_points, args.target_step)
    base = time_to_target(base_points, target)
    print(f"목표 {args.metric} {target:.4f} (기준 run {args.runs[0]}의 step {args.target_step})")
    print(f"{'run':34s} {'name':44s} {'metric':16s} {'step':>7s} {'seconds':>9s} {'speedup':>8s}")
    for run_id in args.runs:
        name = client.get_run(run_id).info.run_name
        metrics = [args.metric]
        if client.get_metric_history(run_id, f"{args.metric}_avg"):
            metrics.append(f"{args.metric}_avg")
        for metric in metrics:
            hit = time_to_target(run_points(client, run_id, metric), target)
            if hit is None:
                print(f"{run_id:34s} {name:44s} {metric:16s} {'—':>7s} {'못 닿음':>9s}")
                continue
            speedup = base.train_seconds / hit.train_seconds
            print(
                f"{run_id:34s} {name:44s} {metric:16s} {hit.step:7d} "
                f"{hit.train_seconds:9.0f} {speedup:7.2f}×"
            )


if __name__ == "__main__":
    main()
