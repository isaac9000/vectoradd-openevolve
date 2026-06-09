# EVOLVE-BLOCK-START
"""
Initial float16 vector addition with Triton kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vecadd_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def custom_kernel(data):
    a, b = data
    c = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    vecadd_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return c
# EVOLVE-BLOCK-END
