from pathlib import Path
import numpy as np

def test_cuda_backproject_box800_complex_addresses_use_64_bit_offsets():
    """The scalar address of a box-800 complex accumulator exceeds int32."""

    accumulator_shape = (1603, 1603, 802)
    spatial_elements = int(np.prod(accumulator_shape, dtype=np.int64))
    last_complex_scalar_offset = (spatial_elements - 1) * 2 + 1
    int32_max = int(np.iinfo(np.int32).max)

    assert spatial_elements == 2_060_826_418
    assert spatial_elements - 1 <= int32_max
    assert last_complex_scalar_offset == 4_121_652_835
    assert last_complex_scalar_offset > int32_max

    cuda_source = Path(__file__).resolve().parents[2] / "recovar" / "cuda" / "cuda_backproject.cu"
    text = cuda_source.read_text()
    helper = text[
        text.index("int64_t volume_spatial_offset(") : text.index("#define BLOCK_SIZE")
    ]
    scatter = text[
        text.index("static __device__ __forceinline__ void scatter_nearest(") :
        text.index("/* Strict RELION x-half diagnostic:")
    ]

    assert "static_cast<int64_t>(i0) * stride0" in helper
    assert "static_cast<int64_t>(i1) * stride1" in helper
    assert scatter.count("const int64_t off") == 8
    assert scatter.count("volume_spatial_offset(") == 8
    assert "const int off =" not in scatter
    indexed_batch = text[
        text.index("batch_backproject_indexed_kernel(") :
        text.index("/* One invocation corresponds to one RELION particle.")
    ]
    start = text.index("batch_backproject_kernel(")
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    dense_batch = text[start:end]
    for batch_scatter in (indexed_batch, dense_batch):
        assert "const int64_t vol_scalar_stride" in batch_scatter
        assert "vol_stride * 2" not in batch_scatter
