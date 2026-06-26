"""Export a small generic DataflowProgram.

This example is intentionally not DNN-specific. It shows the raw portable
schema: objects, reusable compute blocks, ordered tasks, and optional metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataflow_sim.workloads.dataflow import (
    ComputeBlock,
    DataflowCost,
    DataflowMetrics,
    DataflowObject,
    DataflowOutput,
    DataflowProgram,
    DataflowTask,
)


def fixed(name: str, runtime_us: int) -> DataflowCost:
    return DataflowCost(kind="fixed", name=name, runtime_us=runtime_us)


def roofline(name: str, flops: int, memory_bytes: int) -> DataflowCost:
    return DataflowCost(
        kind="roofline",
        name=name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="matmul",
    )


def build_program() -> DataflowProgram:
    preprocess = ComputeBlock(
        key="preprocess",
        name="Preprocess",
        category="input",
        subops=[fixed("decode", 35)],
    )
    projection = ComputeBlock(
        key="projection",
        name="Projection",
        category="encoder",
        subops=[
            roofline("matmul", flops=8_000_000_000, memory_bytes=120_000_000),
            fixed("activation", 12),
        ],
    )
    reduce = ComputeBlock(
        key="reduce",
        name="Reduce",
        category="output",
        subops=[
            DataflowCost(
                kind="roofline",
                name="streaming_reduce",
                flops=0,
                memory_bytes=64_000_000,
                efficiency="memory",
            )
        ],
    )

    return DataflowProgram(
        name="generic-three-task-pipeline",
        description="Small hardware-free dataflow program for import/export demos.",
        metadata={"domain": "example"},
        metrics=DataflowMetrics(primary_unit="items", primary_count=4096),
        objects=[
            DataflowObject(
                id="input_batch",
                size_bytes=16 * 1024 * 1024,
                initial_location="fast",
                role="input",
            ),
            DataflowObject(
                id="projection_weight",
                size_bytes=64 * 1024 * 1024,
                initial_location="backing",
                role="parameter",
            ),
        ],
        compute_blocks=[preprocess, projection, reduce],
        tasks=[
            DataflowTask(
                id="preprocess_0",
                label="Preprocess Batch",
                group="input",
                compute_block_key="preprocess",
                inputs=["input_batch"],
                outputs=[
                    DataflowOutput(
                        id="features",
                        size_bytes=16 * 1024 * 1024,
                        role="activation",
                    )
                ],
            ),
            DataflowTask(
                id="projection_0",
                label="Projection",
                group="encoder",
                compute_block_key="projection",
                inputs=["features", "projection_weight"],
                outputs=[
                    DataflowOutput(
                        id="hidden",
                        size_bytes=16 * 1024 * 1024,
                        role="activation",
                    )
                ],
            ),
            DataflowTask(
                id="reduce_0",
                label="Reduce Output",
                group="output",
                compute_block_key="reduce",
                inputs=["hidden"],
                outputs=[
                    DataflowOutput(
                        id="result",
                        size_bytes=1 * 1024 * 1024,
                        role="output",
                    )
                ],
            ),
        ],
        final_locations={"result": "backing"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("generic_pipeline.dataflow.json"),
        help="Path to write the exported DataflowProgram JSON.",
    )
    args = parser.parse_args()

    program = build_program()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(program.model_dump(mode="json"), indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
