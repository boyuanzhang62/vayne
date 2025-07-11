# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics_e as longbench_scorer_e
from benchmarks.longbenchv2.calculate_metrics import calculate_metrics as longbenchv2_scorer
from benchmarks.loogle.calculate_metrics import calculate_metrics as loogle_scorer
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from benchmarks.zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer

from kvpress import (
    AdaKVPress,
    BlockPress,
    ChunkKVPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    IdlePress,
    ErrorBoundQuantizedPress,
    KiviPress,
    Vayne0Press,
    Vayne1Press,
)

# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "Xnhyacinth/LongBench-v2",
}

SCORER_REGISTRY = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
    "longbench-e": longbench_scorer_e,
    "longbench-v2": longbenchv2_scorer,
}


PRESS_REGISTRY = {
    "ada_expected_attention": AdaKVPress(ExpectedAttentionPress()),
    "ada_expected_attention_e2": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "adasnapkv": AdaKVPress(SnapKVPress()),
    "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "criti_ada_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "criti_adasnapkv": CriticalAdaKVPress(SnapKVPress()),
    "criti_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "criti_snapkv": CriticalKVPress(SnapKVPress()),
    "duo_attention": DuoAttentionPress(),
    "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
    "expected_attention": ExpectedAttentionPress(),
    "finch": FinchPress(),
    "keydiff": KeyDiffPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "pyramidkv": PyramidKVPress(),
    "qfilter": QFilterPress(),
    "random": RandomPress(),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "idle": IdlePress(),
    "error_bound_quantized": ErrorBoundQuantizedPress(),
    "kivi": KiviPress(),
    "vayne0": Vayne0Press(),
    "vayne1": Vayne1Press(),
}
