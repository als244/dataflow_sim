"""Nemotron-style ReLU2 mixture-of-experts module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule, OpDTypePolicy
from dataflow_sim.workloads.modules.nemotron_dimensions import NemotronDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd
from dataflow_sim.workloads.ops import optimizer as opt_ops


class ReLU2MoE(DataflowModule):
    def __init__(self, dims: NemotronDimensions) -> None:
        super().__init__(name="ReLU2MoE")
        self.dims = dims

    def routed_tokens(self, tokens: int) -> int:
        return self.routed_tokens_for_ep(tokens, ep_group_size=1)

    def local_routed_experts(self, *, ep_group_size: int) -> int:
        dims = self.dims
        if dims.num_routed_experts <= 0 or dims.top_k <= 0:
            return 0
        if dims.num_routed_experts % ep_group_size != 0:
            raise ValueError(
                f"ep_group_size={ep_group_size} must divide routed expert count "
                f"{dims.num_routed_experts}"
            )
        return dims.num_routed_experts // ep_group_size

    def routed_tokens_for_ep(self, tokens: int, *, ep_group_size: int) -> int:
        dims = self.dims
        if dims.num_routed_experts <= 0 or dims.top_k <= 0:
            return 0
        self.local_routed_experts(ep_group_size=ep_group_size)
        return tokens * dims.top_k * ep_group_size // dims.num_routed_experts

    def optimizer_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        dims = self.dims
        matrices: list[opt_ops.OptimizerMatrix] = []
        if dims.num_shared_experts > 0:
            matrices.extend(
                [
                    opt_ops.OptimizerMatrix(
                        "shared_mlp_up",
                        dims.d_model,
                        dims.shared_expert_dim,
                        dims.num_shared_experts,
                        True,
                    ),
                    opt_ops.OptimizerMatrix(
                        "shared_mlp_down",
                        dims.shared_expert_dim,
                        dims.d_model,
                        dims.num_shared_experts,
                        True,
                    ),
                ]
            )
        if dims.num_routed_experts > 0 and dims.top_k > 0:
            matrices.extend(
                [
                    opt_ops.OptimizerMatrix(
                        "routed_mlp_up",
                        dims.d_model,
                        dims.expert_dim,
                        dims.num_routed_experts,
                        True,
                        True,
                    ),
                    opt_ops.OptimizerMatrix(
                        "routed_mlp_down",
                        dims.expert_dim,
                        dims.d_model,
                        dims.num_routed_experts,
                        True,
                        True,
                    ),
                ]
            )
        return matrices

    @staticmethod
    def _policy(bytes_per_element: float | OpDTypePolicy) -> OpDTypePolicy:
        return (
            bytes_per_element
            if isinstance(bytes_per_element, OpDTypePolicy)
            else OpDTypePolicy.from_single_bpe(bytes_per_element)
        )

    def forward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        local_routed_experts = self.local_routed_experts(
            ep_group_size=policy.ep_group_size
        )
        routed_tokens = self.routed_tokens_for_ep(
            tokens,
            ep_group_size=policy.ep_group_size,
        )
        routed_total_tokens = routed_tokens * local_routed_experts
        movement_efficiency = "scale_up" if policy.ep_group_size > 1 else "memory"
        ops: list[DataflowCost] = []
        if dims.num_shared_experts > 0:
            ops.append(
                fwd.matmul(
                    "shared_mlp_up",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.shared_expert_dim,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.expert_dispatch_bpe,
                    weight_bytes_per_element=policy.expert_weight_bpe,
                    output_bytes_per_element=policy.activation_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=dims.num_shared_experts,
                )
            )
        if dims.num_routed_experts > 0 and dims.top_k > 0:
            ops.append(
                fwd.scatter(
                    "x_scatter",
                    tokens=tokens,
                    dim=dims.d_model,
                    fanout=dims.top_k,
                    bytes_per_element=policy.activation_bpe,
                    input_bytes_per_element=policy.activation_bpe,
                    output_bytes_per_element=policy.expert_dispatch_bpe,
                    efficiency=movement_efficiency,
                )
            )
            if routed_tokens > 0:
                ops.append(
                    fwd.matmul(
                        "routed_mlp_up_one_expert",
                        tokens=routed_tokens,
                        input_dim=dims.d_model,
                        output_dim=dims.expert_dim,
                        bytes_per_element=policy.activation_bpe,
                        activation_bytes_per_element=policy.expert_dispatch_bpe,
                        weight_bytes_per_element=policy.expert_weight_bpe,
                        output_bytes_per_element=policy.activation_bpe,
                        compute_precision=policy.expert_compute_precision,
                        count=local_routed_experts,
                    )
                )
        if dims.num_shared_experts > 0:
            ops.append(
                fwd.relu2(
                    "shared_relu2",
                    tokens=tokens * dims.num_shared_experts,
                    dim=dims.shared_expert_dim,
                    bytes_per_element=policy.activation_bpe,
                )
            )
        if routed_total_tokens > 0:
            ops.append(
                fwd.relu2(
                    "routed_relu2",
                    tokens=routed_total_tokens,
                    dim=dims.expert_dim,
                    bytes_per_element=policy.activation_bpe,
                )
            )
        if dims.num_shared_experts > 0:
            ops.append(
                fwd.matmul(
                    "shared_mlp_down",
                    tokens=tokens,
                    input_dim=dims.shared_expert_dim,
                    output_dim=dims.d_model,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.activation_bpe,
                    weight_bytes_per_element=policy.expert_weight_bpe,
                    output_bytes_per_element=policy.activation_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=dims.num_shared_experts,
                )
            )
        if dims.num_routed_experts > 0 and dims.top_k > 0:
            if routed_tokens > 0:
                ops.append(
                    fwd.matmul(
                        "routed_mlp_down_one_expert",
                        tokens=routed_tokens,
                        input_dim=dims.expert_dim,
                        output_dim=dims.d_model,
                        bytes_per_element=policy.activation_bpe,
                        activation_bytes_per_element=policy.activation_bpe,
                        weight_bytes_per_element=policy.expert_weight_bpe,
                        output_bytes_per_element=policy.activation_bpe,
                        compute_precision=policy.expert_compute_precision,
                        count=local_routed_experts,
                    )
                )
            ops.append(
                fwd.gather(
                    "x_gather",
                    tokens=tokens,
                    dim=dims.d_model,
                    fanin=dims.top_k,
                    bytes_per_element=policy.activation_bpe,
                    efficiency=movement_efficiency,
                )
            )
        if dims.num_shared_experts > 0 or dims.num_routed_experts > 0:
            ops.append(
                fwd.memory(
                    "moe_combine_residual",
                    bytes_total=4 * tokens * dims.d_model * policy.activation_bpe,
                )
            )
        return ops

    def dgrad_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        local_routed_experts = self.local_routed_experts(
            ep_group_size=policy.ep_group_size
        )
        routed_tokens = self.routed_tokens_for_ep(
            tokens,
            ep_group_size=policy.ep_group_size,
        )
        routed_total_tokens = routed_tokens * local_routed_experts
        movement_efficiency = "scale_up" if policy.ep_group_size > 1 else "memory"
        ops: list[DataflowCost] = []
        if dims.num_routed_experts > 0 and dims.top_k > 0:
            ops.append(
                bwd.scatter_grad(
                    "dy_scatter",
                    tokens=tokens,
                    dim=dims.d_model,
                    fanout=dims.top_k,
                    bytes_per_element=policy.activation_bpe,
                    input_bytes_per_element=policy.activation_bpe,
                    output_bytes_per_element=policy.expert_dispatch_bpe,
                    efficiency=movement_efficiency,
                )
            )
        if routed_tokens > 0:
            ops.append(
                bwd.matmul_input_grad(
                    "routed_mlp_down_one_expert_dgrad",
                    tokens=routed_tokens,
                    input_dim=dims.expert_dim,
                    output_dim=dims.d_model,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.activation_bpe,
                    weight_bytes_per_element=policy.expert_weight_bpe,
                    upstream_gradient_bytes_per_element=policy.expert_dispatch_bpe,
                    input_gradient_bytes_per_element=policy.activation_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=local_routed_experts,
                )
            )
        if dims.num_shared_experts > 0:
            ops.append(
                bwd.matmul_input_grad(
                    "shared_mlp_down_dgrad",
                    tokens=tokens,
                    input_dim=dims.shared_expert_dim,
                    output_dim=dims.d_model,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.activation_bpe,
                    weight_bytes_per_element=policy.expert_weight_bpe,
                    upstream_gradient_bytes_per_element=policy.activation_bpe,
                    input_gradient_bytes_per_element=policy.activation_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=dims.num_shared_experts,
                )
            )
        if dims.num_shared_experts > 0:
            ops.append(
                bwd.relu2_grad(
                    "shared_relu2_bwd",
                    tokens=tokens * dims.num_shared_experts,
                    dim=dims.shared_expert_dim,
                    bytes_per_element=policy.activation_bpe,
                )
            )
        if routed_total_tokens > 0:
            ops.append(
                bwd.relu2_grad(
                    "routed_relu2_bwd",
                    tokens=routed_total_tokens,
                    dim=dims.expert_dim,
                    bytes_per_element=policy.activation_bpe,
                )
            )
        if routed_tokens > 0:
            ops.append(
                bwd.matmul_input_grad(
                    "routed_mlp_up_one_expert_dgrad",
                    tokens=routed_tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.expert_dim,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.activation_bpe,
                    weight_bytes_per_element=policy.expert_weight_bpe,
                    upstream_gradient_bytes_per_element=policy.activation_bpe,
                    input_gradient_bytes_per_element=policy.expert_dispatch_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=local_routed_experts,
                )
            )
        if dims.num_routed_experts > 0 and dims.top_k > 0:
            ops.append(
                bwd.gather_grad(
                    "dy_gather",
                    tokens=tokens,
                    dim=dims.d_model,
                    fanin=dims.top_k,
                    bytes_per_element=policy.activation_bpe,
                    input_bytes_per_element=policy.expert_dispatch_bpe,
                    output_bytes_per_element=policy.activation_bpe,
                    efficiency=movement_efficiency,
                )
            )
        if dims.num_shared_experts > 0:
            ops.append(
                bwd.matmul_input_grad(
                    "shared_mlp_up_dgrad",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.shared_expert_dim,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.activation_bpe,
                    weight_bytes_per_element=policy.expert_weight_bpe,
                    upstream_gradient_bytes_per_element=policy.activation_bpe,
                    input_gradient_bytes_per_element=policy.expert_dispatch_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=dims.num_shared_experts,
                )
            )
        return ops

    def wgrad_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        local_routed_experts = self.local_routed_experts(
            ep_group_size=policy.ep_group_size
        )
        routed_tokens = self.routed_tokens_for_ep(
            tokens,
            ep_group_size=policy.ep_group_size,
        )
        ops: list[DataflowCost] = []
        if routed_tokens > 0:
            ops.append(
                bwd.matmul_weight_grad(
                    "routed_mlp_down_one_expert_wgrad",
                    tokens=routed_tokens,
                    input_dim=dims.expert_dim,
                    output_dim=dims.d_model,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.activation_bpe,
                    upstream_gradient_bytes_per_element=policy.expert_dispatch_bpe,
                    parameter_gradient_bytes_per_element=policy.gradient_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=local_routed_experts,
                )
            )
        if dims.num_shared_experts > 0:
            ops.append(
                bwd.matmul_weight_grad(
                    "shared_mlp_down_wgrad",
                    tokens=tokens,
                    input_dim=dims.shared_expert_dim,
                    output_dim=dims.d_model,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.activation_bpe,
                    upstream_gradient_bytes_per_element=policy.activation_bpe,
                    parameter_gradient_bytes_per_element=policy.gradient_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=dims.num_shared_experts,
                )
            )
        if routed_tokens > 0:
            ops.append(
                bwd.matmul_weight_grad(
                    "routed_mlp_up_one_expert_wgrad",
                    tokens=routed_tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.expert_dim,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.expert_dispatch_bpe,
                    upstream_gradient_bytes_per_element=policy.activation_bpe,
                    parameter_gradient_bytes_per_element=policy.gradient_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=local_routed_experts,
                )
            )
        if dims.num_shared_experts > 0:
            ops.append(
                bwd.matmul_weight_grad(
                    "shared_mlp_up_wgrad",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.shared_expert_dim,
                    bytes_per_element=policy.activation_bpe,
                    activation_bytes_per_element=policy.expert_dispatch_bpe,
                    upstream_gradient_bytes_per_element=policy.activation_bpe,
                    parameter_gradient_bytes_per_element=policy.gradient_bpe,
                    compute_precision=policy.expert_compute_precision,
                    count=dims.num_shared_experts,
                )
            )
        return ops

    def backward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        return self.dgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element) + self.wgrad_ops(
            tokens=tokens,
            bytes_per_element=bytes_per_element,
        )

    def recompute_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        return [
            op.model_copy(update={"effective_flops": 0})
            for op in self.forward_ops(tokens=tokens, bytes_per_element=bytes_per_element)
            if op.name not in {"shared_mlp_down", "routed_mlp_down_one_expert", "x_gather", "moe_combine_residual"}
        ]
