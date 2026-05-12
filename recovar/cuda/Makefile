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

# Detect nvcc major.minor so we can conditionally enable arches.
# Older toolkits (CUDA <12.8) reject `compute_100`/`compute_120` with
# "unsupported gpu architecture". Newer toolkits (CUDA >=13) reject
# `compute_70`/`compute_75` for the same reason — CUDA 13 dropped Volta
# and Turing support entirely.
NVCC_VERSION := $(shell $(NVCC) --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $$2}')
NVCC_MAJOR := $(word 1, $(subst ., ,$(NVCC_VERSION)))
NVCC_MINOR := $(word 2, $(subst ., ,$(NVCC_VERSION)))
# CUDA 12.8+ adds Blackwell (sm_100/sm_120); CUDA 13 drops sm_70/sm_75.
HAS_BLACKWELL := $(shell awk -v M=$(NVCC_MAJOR) -v m=$(NVCC_MINOR) 'BEGIN{print (M>12 || (M==12 && m>=8))?"yes":"no"}')
HAS_VOLTA_TURING := $(shell awk -v M=$(NVCC_MAJOR) 'BEGIN{print (M<13)?"yes":"no"}')

# Multi-arch covering realistic academic cryo-EM hardware. Each line is
# kept on a separate gencode group so we can drop individual arches
# conditionally below without rewriting the whole list.
#   sm_70  — V100, Titan V (Volta)                     [requires nvcc <13]
#   sm_75  — T4, RTX 20-series, Quadro RTX (Turing)    [requires nvcc <13]
#   sm_80  — A100, A30 (Ampere data-center)
#   sm_86  — RTX 30-series, A40, A10 (Ampere consumer/workstation)
#   sm_89  — RTX 40-series, L40, L4 (Ada Lovelace)
#   sm_90  — H100, H200 (Hopper)
#   sm_100 — B100, B200, GB200 (Blackwell data-center) [requires nvcc >=12.8]
#   sm_120 — RTX 50/RTX PRO Blackwell (Blackwell consumer) [requires nvcc >=12.8]
# Plus a PTX fallback so archs not in the SASS list JIT through the driver.
# Pascal (sm_60/61) is opt-in: pre-Volta is increasingly rare; override
# CUDA_ARCH at make time if you need it.
CUDA_ARCH_AMPERE_HOPPER := -gencode arch=compute_80,code=sm_80 \
                           -gencode arch=compute_86,code=sm_86 \
                           -gencode arch=compute_89,code=sm_89 \
                           -gencode arch=compute_90,code=sm_90

ifeq ($(HAS_VOLTA_TURING),yes)
CUDA_ARCH_VOLTA_TURING := -gencode arch=compute_70,code=sm_70 \
                          -gencode arch=compute_75,code=sm_75
else
CUDA_ARCH_VOLTA_TURING :=
endif

ifeq ($(HAS_BLACKWELL),yes)
CUDA_ARCH_BLACKWELL := -gencode arch=compute_100,code=sm_100 \
                       -gencode arch=compute_120,code=sm_120 \
                       -gencode arch=compute_120,code=compute_120
else
# Fall back to compute_90 PTX so the kernel can still JIT to Blackwell on
# the user's machine even when the build toolkit is too old to emit native
# Blackwell SASS. Driver-side JIT from compute_90 PTX to sm_100/sm_120 is
# imperfect but better than no fallback at all.
CUDA_ARCH_BLACKWELL := -gencode arch=compute_90,code=compute_90
endif

CUDA_ARCH ?= $(CUDA_ARCH_VOLTA_TURING) $(CUDA_ARCH_AMPERE_HOPPER) $(CUDA_ARCH_BLACKWELL)

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
