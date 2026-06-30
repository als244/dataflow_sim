"""Backward matmul op helpers."""
from __future__ import annotations

import math

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import OpDTypePolicy
from dataflow_sim.workloads.ops._common import matmul_efficiency, roofline


def matmul_input_grad(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: float | OpDTypePolicy = 2,
    activation_bytes_per_element: float | None = None,
    weight_bytes_per_element: float | None = None,
    gradient_bytes_per_element: float | None = None,
    upstream_gradient_bytes_per_element: float | None = None,
    input_gradient_bytes_per_element: float | None = None,
    count: int = 1,
    memory_bytes: int | None = None,
    compute_precision: str = "bf16",
) -> DataflowCost:
    if isinstance(bytes_per_element, OpDTypePolicy):
        policy = bytes_per_element
        bytes_per_element = policy.activation_bpe
        activation_bytes_per_element = policy.activation_bpe
        weight_bytes_per_element = policy.weight_bpe
        upstream_gradient_bytes_per_element = policy.activation_bpe
        input_gradient_bytes_per_element = policy.activation_bpe
        compute_precision = policy.compute_precision
    if gradient_bytes_per_element is not None:
        upstream_gradient_bytes_per_element = (
            gradient_bytes_per_element
            if upstream_gradient_bytes_per_element is None
            else upstream_gradient_bytes_per_element
        )
        input_gradient_bytes_per_element = (
            gradient_bytes_per_element
            if input_gradient_bytes_per_element is None
            else input_gradient_bytes_per_element
        )
    return _matmul_grad(
        name,
        tokens=tokens,
        input_dim=input_dim,
        output_dim=output_dim,
        bytes_per_element=bytes_per_element,
        activation_bytes_per_element=activation_bytes_per_element,
        weight_bytes_per_element=weight_bytes_per_element,
        upstream_gradient_bytes_per_element=upstream_gradient_bytes_per_element,
        input_gradient_bytes_per_element=input_gradient_bytes_per_element,
        parameter_gradient_bytes_per_element=None,
        count=count,
        accumulator_elements=0,
        memory_bytes=memory_bytes,
        mode="dgrad",
        compute_precision=compute_precision,
    )


def matmul_weight_grad(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: float | OpDTypePolicy = 2,
    activation_bytes_per_element: float | None = None,
    weight_bytes_per_element: float | None = None,
    gradient_bytes_per_element: float | None = None,
    upstream_gradient_bytes_per_element: float | None = None,
    parameter_gradient_bytes_per_element: float | None = None,
    count: int = 1,
    accumulate: bool = True,
    memory_bytes: int | None = None,
    compute_precision: str = "bf16",
) -> DataflowCost:
    if isinstance(bytes_per_element, OpDTypePolicy):
        policy = bytes_per_element
        bytes_per_element = policy.activation_bpe
        activation_bytes_per_element = policy.activation_bpe
        weight_bytes_per_element = policy.weight_bpe
        upstream_gradient_bytes_per_element = policy.activation_bpe
        parameter_gradient_bytes_per_element = policy.gradient_bpe
        compute_precision = policy.compute_precision
    if gradient_bytes_per_element is not None:
        upstream_gradient_bytes_per_element = (
            gradient_bytes_per_element
            if upstream_gradient_bytes_per_element is None
            else upstream_gradient_bytes_per_element
        )
        parameter_gradient_bytes_per_element = (
            gradient_bytes_per_element
            if parameter_gradient_bytes_per_element is None
            else parameter_gradient_bytes_per_element
        )
    return _matmul_grad(
        name,
        tokens=tokens,
        input_dim=input_dim,
        output_dim=output_dim,
        bytes_per_element=bytes_per_element,
        activation_bytes_per_element=activation_bytes_per_element,
        weight_bytes_per_element=weight_bytes_per_element,
        upstream_gradient_bytes_per_element=upstream_gradient_bytes_per_element,
        input_gradient_bytes_per_element=None,
        parameter_gradient_bytes_per_element=parameter_gradient_bytes_per_element,
        count=count,
        accumulator_elements=input_dim * output_dim if accumulate else 0,
        memory_bytes=memory_bytes,
        mode="wgrad",
        compute_precision=compute_precision,
    )


def _matmul_grad(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: float,
    activation_bytes_per_element: float | None,
    weight_bytes_per_element: float | None,
    upstream_gradient_bytes_per_element: float | None,
    input_gradient_bytes_per_element: float | None,
    parameter_gradient_bytes_per_element: float | None,
    count: int,
    accumulator_elements: int,
    memory_bytes: int | None,
    mode: str,
    compute_precision: str,
) -> DataflowCost:
    flops = 2 * tokens * input_dim * output_dim
    bytes_total = memory_bytes
    if bytes_total is None:
        act_bpe = bytes_per_element if activation_bytes_per_element is None else activation_bytes_per_element
        weight_bpe = bytes_per_element if weight_bytes_per_element is None else weight_bytes_per_element
        upstream_grad_bpe = (
            bytes_per_element
            if upstream_gradient_bytes_per_element is None
            else upstream_gradient_bytes_per_element
        )
        if mode == "dgrad":
            input_grad_bpe = (
                upstream_grad_bpe
                if input_gradient_bytes_per_element is None
                else input_gradient_bytes_per_element
            )
            bytes_total = math.ceil(
                tokens * output_dim * upstream_grad_bpe
                + input_dim * output_dim * weight_bpe
                + tokens * input_dim * input_grad_bpe
            )
        else:
            param_grad_bpe = (
                upstream_grad_bpe
                if parameter_gradient_bytes_per_element is None
                else parameter_gradient_bytes_per_element
            )
            bytes_total = math.ceil(
                tokens * input_dim * act_bpe
                + tokens * output_dim * upstream_grad_bpe
                + input_dim * output_dim * param_grad_bpe
                + accumulator_elements * param_grad_bpe
            )
    return roofline(
        name,
        flops=flops,
        memory_bytes=bytes_total,
        efficiency=matmul_efficiency(compute_precision),
        count=count,
    )
