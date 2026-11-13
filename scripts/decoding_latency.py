#!/usr/bin/env python3
"""Measure PyMatching decoding latency on Stim surface-code memory circuits.

One run decodes `--shots` syndromes of a rotated surface-code memory experiment
(distance d, d rounds of syndrome extraction, uniform circuit-level noise p) one
shot at a time and records the wall-clock time of every decode call.

Reported per distance, in microseconds: mean, standard deviation, median, 99th
and 99.9th percentile, maximum. Two cross-checks are included:
  * batch_mean_us: total time of decode_batch over the same syndromes divided by
    the number of shots (no per-call Python overhead);
  * call_overhead_us: median time of decoding an all-zero syndrome.

The JSON output keeps the keys of results/figure14_p003_1us/latencies_measured.json
(d, mean_decoding_us, p99_9_decoding_us) so existing scripts can read it.

RQ3 setting (Sec. 6.3):
  python decoding_latency.py --distance 21 --p 1e-4 --shots 10000000 \\
      --out ../results/rq3/latency_d21_p1e-4.json

Needs: pip install stim pymatching numpy. Run it alone on one pinned core, e.g.
  taskset -c 2 python decoding_latency.py ...
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pymatching
import stim


def build(distance: int, rounds: int, p: float, basis: str) -> tuple[stim.Circuit, pymatching.Matching]:
    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    return circuit, pymatching.Matching.from_detector_error_model(dem)


def measure(distance: int, rounds: int, p: float, basis: str, shots: int, chunk: int, seed: int) -> dict:
    circuit, matcher = build(distance, rounds, p, basis)
    sampler = circuit.compile_detector_sampler(seed=seed)
    clock = time.perf_counter_ns

    # Warm-up, and the cost of one decode call on an empty syndrome.
    zero = np.zeros(circuit.num_detectors, dtype=np.uint8)
    overhead = np.empty(20000, dtype=np.int64)
    for i in range(overhead.size):
        t0 = clock()
        matcher.decode(zero)
        overhead[i] = clock() - t0

    times = np.empty(shots, dtype=np.int64)  # ns per shot
    batch_ns = 0
    defects = 0
    failures = 0
    done = 0
    while done < shots:
        n = min(chunk, shots - done)
        dets, obs = sampler.sample(n, separate_observables=True)
        dets = np.ascontiguousarray(dets, dtype=np.uint8)
        defects += int(dets.sum())
        out = times[done : done + n]
        for i in range(n):
            syndrome = dets[i]
            t0 = clock()
            matcher.decode(syndrome)
            out[i] = clock() - t0
        t0 = clock()
        predicted = matcher.decode_batch(dets)
        batch_ns += clock() - t0
        failures += int(np.any(predicted != obs, axis=1).sum())
        done += n
        print(f"  d={distance}: {done}/{shots} shots, running mean {times[:done].mean() / 1e3:.3f} us", flush=True)

    us = times / 1e3
    return {
        "d": distance,
        "rounds": rounds,
        "p": p,
        "basis": basis,
        "shots": shots,
        "detectors": circuit.num_detectors,
        "mean_defects_per_shot": defects / shots,
        "logical_failures": failures,
        "mean_decoding_us": float(us.mean()),
        "std_decoding_us": float(us.std()),
        "p50_decoding_us": float(np.percentile(us, 50)),
        "p99_decoding_us": float(np.percentile(us, 99)),
        "p99_9_decoding_us": float(np.percentile(us, 99.9)),
        "max_decoding_us": float(us.max()),
        "batch_mean_us": batch_ns / shots / 1e3,
        "call_overhead_us": float(np.median(overhead) / 1e3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=int, nargs="+", default=[21])
    ap.add_argument("--rounds", type=int, help="syndrome-extraction rounds (default: d)")
    ap.add_argument("--p", type=float, default=1e-4, help="physical error rate of every noise channel")
    ap.add_argument("--basis", choices=["z", "x"], default="z")
    ap.add_argument("--shots", type=int, default=10_000_000)
    ap.add_argument("--chunk", type=int, default=20_000, help="shots sampled at a time (bounds memory)")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results = []
    for d in args.distance:
        result = measure(d, args.rounds or d, args.p, args.basis, args.shots, args.chunk, args.seed + d)
        results.append(result)
        print(
            f"d={d} p={args.p:g}: mean {result['mean_decoding_us']:.3f} us, "
            f"99.9% {result['p99_9_decoding_us']:.3f} us, batch mean {result['batch_mean_us']:.3f} us, "
            f"call overhead {result['call_overhead_us']:.3f} us"
        )
    payload = {
        "status": "passed",
        "shots_total": args.shots * len(args.distance),
        "stim": stim.__version__,
        "pymatching": pymatching.__version__,
        "machine": platform.platform(),
        "processor": platform.processor(),
        "seed": args.seed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
