"""探测 InternData-N1 数据集中各 trajectory 的 ``observation.camera_intrinsic``。

复用 ``exp_scale_compare.py`` 已有的:
  - ``InternNavSequenceLoader`` 统一发现各 trajectory(支持 single trajectory /
    group / 多 group 嵌套);
  - ``_load_intrinsics_from_parquet`` 从 parquet 第一行还原 (3,3) K。

落盘两份产物方便事后核对:
  - ``<out>.json``: 含每个 trajectory 的 K、首末行一致性、行数,以及 fx/fy/cx/cy
    跨 trajectory 的统计(min/median/max);
  - ``<out>.csv``: 一行一个 trajectory,字段同 entries,便于表格查看。

CLI::

    python tools/exp_scale/probe_n1_intrinsics.py <dataset_root> [--out <json_path>]

默认 ``--out=<dataset_root>/.n1_intrinsics_probe.json``,同目录另写一份 ``.csv``。
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from L3ROcc.dataset.intern_nav_adapter import InternNavSequenceLoader  # noqa: E402


def _load_intrinsic_rows(parquet_path):
    """读 parquet 的 ``observation.camera_intrinsic`` 整列;返回 (n_rows, K_first, first_eq_last)。

    K_first 为 (3,3) np.float64 或 None;若该列缺失/类型异常,返回 (0, None, False)。
    """
    if not parquet_path or not os.path.isfile(parquet_path):
        return 0, None, False
    try:
        df = pd.read_parquet(parquet_path, columns=["observation.camera_intrinsic"])
    except Exception:
        return 0, None, False
    if "observation.camera_intrinsic" not in df.columns or len(df) == 0:
        return 0, None, False

    def _to_3x3(raw):
        try:
            arr = np.asarray(
                raw.tolist() if hasattr(raw, "tolist") else raw, dtype=np.float64
            )
        except Exception:
            return None
        if arr.shape == (3, 3):
            return arr
        if arr.size == 9:
            return arr.reshape(3, 3)
        return None

    k_first = _to_3x3(df["observation.camera_intrinsic"].iloc[0])
    k_last = _to_3x3(df["observation.camera_intrinsic"].iloc[-1])
    if k_first is None:
        return len(df), None, False
    first_eq_last = (
        k_last is not None
        and k_first.shape == k_last.shape
        and np.allclose(k_first, k_last)
    )
    return len(df), k_first, bool(first_eq_last)


def _agg_stats(values):
    """跨 trajectory 聚合一个标量,返回 None 当全部缺失。"""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def probe(dataset_root):
    """遍历 dataset_root,逐 trajectory 读 K,返回 dict 结构(供 JSON 落盘)。"""
    loader = InternNavSequenceLoader(dataset_root)
    if len(loader) == 0:
        return {
            "dataset_root": dataset_root,
            "n_trajectories": 0,
            "stats": {},
            "entries": [],
        }

    entries = []
    fx_list, fy_list, cx_list, cy_list = [], [], [], []
    n_with_k = 0

    for i in range(len(loader)):
        traj_dir = loader.trajectory_dirs[i]
        parquet_path = loader.trajectory_data_paths[i]
        video_path = loader.trajectory_video_paths[i]
        n_rows, k, first_eq_last = _load_intrinsic_rows(parquet_path)

        entry = {
            "trajectory_dir": traj_dir,
            "parquet": parquet_path,
            "video": video_path,
            "n_rows": int(n_rows),
            "K_first_row_eq_last": first_eq_last,
            "K": k.tolist() if k is not None else None,
        }
        if k is not None:
            n_with_k += 1
            fx_list.append(float(k[0, 0]))
            fy_list.append(float(k[1, 1]))
            cx_list.append(float(k[0, 2]))
            cy_list.append(float(k[1, 2]))
        entries.append(entry)

    stats = {
        "n_with_K": n_with_k,
        "n_missing_K": len(entries) - n_with_k,
        "fx": _agg_stats(fx_list),
        "fy": _agg_stats(fy_list),
        "cx": _agg_stats(cx_list),
        "cy": _agg_stats(cy_list),
    }

    return {
        "dataset_root": dataset_root,
        "n_trajectories": len(entries),
        "stats": stats,
        "entries": entries,
    }


def _write_csv(csv_path, entries):
    """与 JSON 同名的 .csv;一行一 trajectory,K 拆成 fx/fy/cx/cy 四列方便 grep。"""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["trajectory_dir", "parquet", "n_rows", "fx", "fy", "cx", "cy",
             "K_first_row_eq_last", "has_K"]
        )
        for e in entries:
            k = e["K"]
            if k is None:
                w.writerow([e["trajectory_dir"], e["parquet"], e["n_rows"],
                            "", "", "", "", e["K_first_row_eq_last"], False])
            else:
                w.writerow([e["trajectory_dir"], e["parquet"], e["n_rows"],
                            k[0][0], k[1][1], k[0][2], k[1][2],
                            e["K_first_row_eq_last"], True])


def main():
    ap = argparse.ArgumentParser(
        description="探测 InternData-N1 数据集每个 trajectory 的 observation.camera_intrinsic。"
    )
    ap.add_argument("dataset_root", type=str, help="N1 数据根目录(单 trajectory / group / 多 group)")
    ap.add_argument(
        "--out", type=str, default="",
        help="JSON 输出路径;默认 <dataset_root>/.n1_intrinsics_probe.json(同目录写 .csv)",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"[error] dataset_root 不存在或不是目录: {args.dataset_root}", file=sys.stderr)
        sys.exit(1)

    out_json = args.out or os.path.join(args.dataset_root, ".n1_intrinsics_probe.json")
    out_csv = os.path.splitext(out_json)[0] + ".csv"

    report = probe(args.dataset_root)

    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_csv(out_csv, report["entries"])

    n = report["n_trajectories"]
    s = report["stats"]
    print(f"[probe] dataset_root = {args.dataset_root}")
    print(f"        trajectories = {n}  (with_K={s.get('n_with_K', 0)}, "
          f"missing_K={s.get('n_missing_K', 0)})")
    for key in ("fx", "fy", "cx", "cy"):
        v = s.get(key)
        if v is not None:
            print(f"        {key}: median={v['median']:.4f}  "
                  f"[min={v['min']:.4f}, max={v['max']:.4f}]  n={v['n']}")
    print(f"[probe] JSON -> {out_json}")
    print(f"[probe] CSV  -> {out_csv}")


if __name__ == "__main__":
    main()
