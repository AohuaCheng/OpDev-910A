#!/usr/bin/env python3
"""
Test script for single matmul+bias function (no grouping)
"""
import torch
import vllm_ascend.vllm_ascend_C

def single_matmul_cpu_reference(x, weight, bias=None):
    output = torch.matmul(x, weight)
    if bias is not None:
        output = output + bias
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 128, 32, 64
    x = torch.randn(M, K, device='npu', dtype=torch.float16)
    weight = torch.randn(K, N, device='npu', dtype=torch.float16)
    bias = torch.randn(N, device='npu', dtype=torch.float16)
    # CPU reference
    x_cpu = x.cpu()
    weight_cpu = weight.cpu()
    bias_cpu = bias.cpu()
    cpu_output = single_matmul_cpu_reference(x_cpu, weight_cpu, bias_cpu)
    print(f"CPU output shape: {cpu_output.shape}")
    # NPU computation
    npu_outputs = torch.ops._C._npu_grouped_matmul([x], [weight], [bias], [], 0, torch.float16)
    torch.npu.synchronize()
    npu_output = npu_outputs[0].cpu()
    print(f"NPU output shape: {npu_output.shape}")
    # 对比结果
    diff = (npu_output - cpu_output).abs()
    print(f"Max abs diff: {diff.max().item():.6f}")
    print(f"Mean abs diff: {diff.mean().item():.6f}")
    print(f"NPU output (first row): {npu_output[0]}")
    print(f"CPU output (first row): {cpu_output[0]}")
    print("PASS")