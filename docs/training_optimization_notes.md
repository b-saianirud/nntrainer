# Training Optimization Analysis & Notes

## Current State (as of 2026-08-14)

### W4A16 Training Path (implemented)

**What was done:**
- `GgmlQuantizer::dequantize()` extended to support FP16 output (previously FP32-only)
- `fc_layer.cpp::forwarding()` and `calcDerivative()` dequantize Q4_0/Q6_K weights
  to the **activation dtype** (FP16 when `ENABLE_FP16`, FP32 otherwise)
- Weight caching: dequantized weight is computed once on first `forwarding()` call,
  then reused for all subsequent forward/backward passes (member `cached_weight_`)

**Why W4A16, not W4A8?**
The existing fused W4A8 kernel (`nntr_gemm_q4_0_4x8_q8_0`) quantizes activations
to Q8_0 on-the-fly inside the GEMM. This adds quantization noise to the activations
on every forward pass, which prevents training convergence. W4A16 (dequantize 4-bit
weight to FP16, keep FP16 activations) is exact for the frozen weight and preserves
full-precision activations — no noise.

**How it works:**
```
Q4_0 bin file (4-bit, ~150MB on disk)
    │
    ▼  (first forward pass only)
GgmlQuantizer::dequantize(weight, act_dtype)
    │  1. Q4_0Utils::dequantizeQ4_0x4/x8 → FP32
    │  2. clone(FP16) if act_dtype == FP16
    ▼
cached_weight_ (FP16, ~300MB in RAM, persistent)
    │
    ▼  (every forward/backward pass)
input_ (FP16) × cached_weight_ (FP16) → hidden_ (FP16)
    ← standard BLAS GEMM (cblas)
```

**Why not just use an FP32 bin file?**
- On-disk: Q4_0 is 4× smaller (150MB vs 600MB for 0.6B model)
- In-RAM: Q4_0 (150MB) + FP16 cache (300MB) = 450MB < FP32 (600MB)
- If Q4_0 is munmap'd after caching: only 300MB RAM (2× less than FP32)
- The cache is FP16 (2 bytes/weight), not FP32 (4 bytes/weight)

**Inference path (unchanged):**
- `incremental_forwarding()` still uses `input_step.dot(weight)` directly
- Dispatches to fused W4A8 kernel (`nntr_gemm_q4_0_4x8_q8_0`)
- Q4_0 weight × Q8_0 activation (quantized on-the-fly)
- Faster (no dequantization copy) but adds Q8_0 quantization noise
- Acceptable for inference (no gradients), not for training



---

## Attention Calculation — Current State

### Training Forward (`trainForwarding`, mha_core.cpp:1645-1758)

**CRITICAL: Entirely scalar, no BLAS, no threading, FP32-only.**

```cpp
// Q × K^T — scalar dot product, no BLAS
for (unsigned int j = 0; j <= i; ++j) {
    for (unsigned int d = 0; d < head_dim; ++d)
        s += q_row[d] * k_row[d];
}
// attn × V — scalar FMA, no BLAS
for (unsigned int j = 0; j <= i; ++j) {
    for (unsigned int d = 0; d < head_dim; ++d)
        out_row[d] += w * v_row[d];
}
```

**Issues:**
1. No BLAS — Q×K^T and attn×V are batched GEMMs, should use `cblas_sgemm`
2. No threading — single thread for all batch×head×seq loops
3. FP32-only — hardcoded check at line 1666 rejects FP16
4. No flash attention — full seq×seq attention matrix materialized
5. No SIMD — scalar FP32 FMAs

### Training Backward (`calcDerivative`, mha_core.cpp:1800-1916)

**Same issues as forward: scalar, no BLAS, no threading, FP32-only.**

Additional issue: `value.clone()` at line 1817 copies V every step.

### Inference Attention (well-optimized)

- `ThreadManager::parallel_for` over KV heads (decode) or query rows (prefill)
- FP16 kernels: `compute_kcaches<_FP16>`, `compute_fp16vcache_transposed`
- Tiled SIMD kernels (`tile_size=4`)
- FP16 KV cache on Android

---

## Hardware Utilization Summary

| Component | BLAS? | Threaded? | SIMD? | FP16? |
|-----------|-------|-----------|-------|-------|
| FC forward (training) | ✅ | via BLAS | ✅ | ✅ |
| FC backward (training) | ✅ | via BLAS | ✅ | ✅ |
| **Attention forward (training)** | ❌ | ❌ | ❌ | ❌ |
| **Attention backward (training)** | ❌ | ❌ | ❌ | ❌ |
| Attention forward (inference) | N/A | ✅ | ✅ | ✅ |
| RoPE (training) | N/A | ❌ | ❌ | partial |
| LoRA GEMMs | ✅ | via BLAS | ✅ | ❌ (FP32) |

---

## Recommended Optimizations (priority order)

### 1. Replace training attention with BLAS GEMM (highest impact)

Q×K^T is a batched GEMM: for each head, `C[i,j] = sum_d Q[i,d] * K[j,d]`.
attn×V is also a batched GEMM: `O[i,d] = sum_j attn[i,j] * V[j,d]`.

Use `cblas_sgemm` in a loop over heads, or `cblas_sgemm_batch` if available.
Expected speedup: 10-50× on attention forward.

### 2. Thread the training attention

Parallelize over heads using `ThreadManager::parallel_for(0, num_heads_Q, ...)`.
The inference path already does this.

### 3. Replace training attention backward with BLAS GEMM

dQ = d_attn × K, dK = d_attn^T × Q, dV = dy^T × attn are all GEMMs.

### 4. Remove `value.clone()` in calcDerivative

Use a const reference or snapshot only needed rows.

### 5. Enable FP16 for training attention

Remove the FP32-only check. Use FP16 GEMMs when model is FP16.

### 6. Thread the RoPE

Parallelize `apply_rotary_emb_tensor_v2` over batch × heads.

### 7. Fused W4A16 GEMM kernel

Currently: dequantize Q4_0 → FP16, then BLAS GEMM.
A fused kernel (like W4A8 but reading FP16 activations) would eliminate the
dequantized weight copy entirely and fuse the unpack into the GEMM.

### 8. LoRA GEMMs in FP16

LoRA adapter weights (loraA, loraB) are FP32. If FP16, the LoRA GEMMs
would be 2× faster. Optimizer would need FP32 master weights (mixed-precision).

### 9. Threaded dequantization

`Q4_0Utils::dequantizeQ4_0x4/x8` is single-threaded. With weight caching
this only runs once, but parallelizing it (OpenMP) would speed up first step.
