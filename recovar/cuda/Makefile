PYTHON     ?= python
NVCC       ?= /usr/local/cuda/bin/nvcc
XLA_INC    := $(shell $(PYTHON) -c "import jax.ffi; print(jax.ffi.include_dir())")

# Multi-arch: A100 (sm_80), RTX 3090 (sm_86), RTX 4090 (sm_89), H100 (sm_90)
CUDA_ARCH  ?= -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89 \
              -gencode arch=compute_90,code=sm_90

NVCC_FLAGS := -O3 -std=c++17 -Xcompiler -fPIC --shared $(CUDA_ARCH) -I$(XLA_INC)

LIB := libcuda_backproject.so

all: $(LIB)

$(LIB): cuda_backproject.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(LIB)

.PHONY: all clean
