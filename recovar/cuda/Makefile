PYTHON     ?= python
NVCC_FROM_PATH := $(shell command -v nvcc 2>/dev/null)
NVCC_FROM_LOCAL_CUDA_PATH := $(shell [ -n "$(LOCAL_CUDA_PATH)" ] && [ -x "$(LOCAL_CUDA_PATH)/bin/nvcc" ] && printf '%s' "$(LOCAL_CUDA_PATH)/bin/nvcc")
NVCC_FROM_CUDA_HOME := $(shell [ -n "$(CUDA_HOME)" ] && [ -x "$(CUDA_HOME)/bin/nvcc" ] && printf '%s' "$(CUDA_HOME)/bin/nvcc")
NVCC_FROM_CUDA_PATH := $(shell [ -n "$(CUDA_PATH)" ] && [ -x "$(CUDA_PATH)/bin/nvcc" ] && printf '%s' "$(CUDA_PATH)/bin/nvcc")

# Precedence:
#   1. NVCC (command line or environment)
#   2. CUDACXX
#   3. nvcc on PATH
#   4. LOCAL_CUDA_PATH/bin/nvcc
#   5. CUDA_HOME/bin/nvcc
#   6. CUDA_PATH/bin/nvcc
ifeq ($(origin NVCC), undefined)
NVCC := $(or \
          $(strip $(CUDACXX)), \
          $(strip $(NVCC_FROM_PATH)), \
          $(strip $(NVCC_FROM_LOCAL_CUDA_PATH)), \
          $(strip $(NVCC_FROM_CUDA_HOME)), \
          $(strip $(NVCC_FROM_CUDA_PATH)) \
        )
endif

XLA_INC    := $(shell $(PYTHON) -c "import jax.ffi; print(jax.ffi.include_dir())")

# Multi-arch covering realistic academic cryo-EM hardware:
#   sm_70 — V100, Titan V (Volta)
#   sm_75 — T4, RTX 20-series, Quadro RTX (Turing)
#   sm_80 — A100, A30 (Ampere data-center)
#   sm_86 — RTX 30-series, A40, A10 (Ampere consumer/workstation)
#   sm_89 — RTX 40-series, L40, L4 (Ada Lovelace)
#   sm_90 — H100, H200 (Hopper)
# Plus a compute_75 PTX target so future archs (sm_100+) JIT forward-compatibly.
# Pascal (sm_60/61) is opt-in: pre-Volta GPUs are increasingly rare in cryo-EM
# clusters; users who need them can override CUDA_ARCH at make time.
CUDA_ARCH  ?= -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89 \
              -gencode arch=compute_90,code=sm_90 \
              -gencode arch=compute_75,code=compute_75

NVCC_FLAGS := -O3 -std=c++17 -Xcompiler -fPIC --shared $(CUDA_ARCH) -I$(XLA_INC)

LIB ?= libcuda_backproject.so

all: $(LIB)

.PHONY: all clean check-nvcc

check-nvcc:
	@if [ -z "$(strip $(NVCC))" ]; then \
		echo "Could not find nvcc. Set NVCC or CUDACXX, activate a CUDA toolkit so nvcc is on PATH, or set LOCAL_CUDA_PATH/CUDA_HOME/CUDA_PATH." >&2; \
		exit 1; \
	fi

$(LIB): cuda_backproject.cu | check-nvcc
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(LIB)
