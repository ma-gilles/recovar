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

# Detect nvcc major.minor so we can conditionally enable newer arches.
# Older toolkits (e.g. CUDA 12.4) reject `compute_100` / `compute_120` with
# "unsupported gpu architecture", so we only request them when supported.
NVCC_VERSION := $(shell $(NVCC) --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $$2}')
NVCC_MAJOR := $(word 1, $(subst ., ,$(NVCC_VERSION)))
NVCC_MINOR := $(word 2, $(subst ., ,$(NVCC_VERSION)))
HAS_BLACKWELL := $(shell awk -v M=$(NVCC_MAJOR) -v m=$(NVCC_MINOR) 'BEGIN{print (M>12 || (M==12 && m>=8))?"yes":"no"}')

# Multi-arch covering realistic academic cryo-EM hardware:
#   sm_70  — V100, Titan V (Volta)
#   sm_75  — T4, RTX 20-series, Quadro RTX (Turing)
#   sm_80  — A100, A30 (Ampere data-center)
#   sm_86  — RTX 30-series, A40, A10 (Ampere consumer/workstation)
#   sm_89  — RTX 40-series, L40, L4 (Ada Lovelace)
#   sm_90  — H100, H200 (Hopper)
#   sm_100 — B100, B200, GB200 (Blackwell data-center, requires nvcc >= 12.8)
#   sm_120 — RTX 50-series, RTX PRO Blackwell (Blackwell consumer, requires nvcc >= 12.8)
# Plus a compute_120 PTX target (or compute_90 when nvcc < 12.8) so future
# archs JIT forward-compatibly through the driver.
# Pascal (sm_60/61) is opt-in: pre-Volta GPUs are increasingly rare in cryo-EM
# clusters; users who need them can override CUDA_ARCH at make time.
CUDA_ARCH_BASE := -gencode arch=compute_70,code=sm_70 \
                  -gencode arch=compute_75,code=sm_75 \
                  -gencode arch=compute_80,code=sm_80 \
                  -gencode arch=compute_86,code=sm_86 \
                  -gencode arch=compute_89,code=sm_89 \
                  -gencode arch=compute_90,code=sm_90

ifeq ($(HAS_BLACKWELL),yes)
CUDA_ARCH ?= $(CUDA_ARCH_BASE) \
             -gencode arch=compute_100,code=sm_100 \
             -gencode arch=compute_120,code=sm_120 \
             -gencode arch=compute_120,code=compute_120
else
# Fall back to compute_90 PTX so the kernel can still JIT to Blackwell on
# the user's machine even when the build toolkit is too old to emit native
# Blackwell SASS. Driver-side JIT from compute_90 PTX to sm_100/sm_120 is
# imperfect but better than no fallback at all.
CUDA_ARCH ?= $(CUDA_ARCH_BASE) \
             -gencode arch=compute_90,code=compute_90
endif

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
