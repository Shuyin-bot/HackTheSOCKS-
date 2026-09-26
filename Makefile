# ── SO-101 / LeRobot dev environment ─────────────────────────────────────────
# Full stack (arm machine):  make          (Linux or Apple-Silicon mac)
# Vision/logic only:         make lite      (Intel mac, or no arm attached)
# See `make help` for all targets. First `make` downloads ~2.5 GB.

CONDA_ENV    ?= lerobot
LITE_ENV     ?= sockvision
PY_VERSION   ?= 3.12
LEROBOT_REPO ?= https://github.com/huggingface/lerobot.git
LEROBOT_DIR  ?= lerobot
# Pin the whole team to one commit (LeRobot v0.6.1):
LEROBOT_REF  ?= 1bc0bdfb20ad4f4f76dc8a68a0e0746d54f50de9
# Extras: motors (feetech) + record/replay/calibrate + dataset(video) + viz
EXTRAS       ?= feetech,core_scripts

CONDA      := $(shell command -v mamba 2>/dev/null || command -v conda 2>/dev/null)
CONDA_BASE := $(shell conda info --base 2>/dev/null)
ENV_PY     := $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/python
LITE_PY    := $(CONDA_BASE)/envs/$(LITE_ENV)/bin/python
UNAME_S    := $(shell uname -s)
UNAME_M    := $(shell uname -m)

.DEFAULT_GOAL := setup
.PHONY: setup lite check-conda guard-arch env ffmpeg clone install verify verify-lite \
        info find-port find-cameras bootstrap apt-deps clean clean-env help

setup: check-conda guard-arch env ffmpeg clone install verify ## Full setup (arm machine)
	@echo ""
	@echo "✅ Done. Next:  conda activate $(CONDA_ENV)"
	@echo "   Then:   make find-port   |   make find-cameras"

lite: check-conda ## Vision/logic env only — works on Intel mac, no arm/torch/lerobot
	@if [ -d "$(CONDA_BASE)/envs/$(LITE_ENV)" ]; then \
	  echo "✓ lite env '$(LITE_ENV)' already exists"; \
	else \
	  echo "→ creating lite env '$(LITE_ENV)' (python $(PY_VERSION))"; \
	  $(CONDA) create -y -n $(LITE_ENV) python=$(PY_VERSION); \
	fi
	@echo "→ installing opencv-python numpy requests (add onnxruntime for CLIP-style matching)"
	$(LITE_PY) -m pip install opencv-python numpy requests
	@$(MAKE) --no-print-directory verify-lite
	@echo "✅ Lite env ready.  conda activate $(LITE_ENV)"

check-conda:
	@if [ -z "$(CONDA)" ]; then \
	  echo "❌ conda/mamba not found."; \
	  echo "   Run 'make bootstrap' to install Miniforge, or install it manually:"; \
	  echo "   https://github.com/conda-forge/miniforge#install"; \
	  exit 1; \
	fi

guard-arch: ## Fail early + clearly on Intel macOS (no torch>=2.7 / TorchCodec there)
	@if [ "$(UNAME_S)" = "Darwin" ] && [ "$(UNAME_M)" = "x86_64" ]; then \
	  echo "❌ Intel (x86_64) macOS detected."; \
	  echo "   LeRobot needs torch>=2.7 (no Intel-mac build) and TorchCodec is unavailable here."; \
	  echo "   The arm/policy stack can't run. Instead run:  make lite"; \
	  echo "   (opencv + numpy + requests — everything the sock-sorter vision needs)."; \
	  exit 1; \
	fi

env: check-conda ## Create the full conda env if missing
	@if [ -d "$(CONDA_BASE)/envs/$(CONDA_ENV)" ]; then \
	  echo "✓ conda env '$(CONDA_ENV)' already exists"; \
	else \
	  echo "→ creating conda env '$(CONDA_ENV)' (python $(PY_VERSION))"; \
	  $(CONDA) create -y -n $(CONDA_ENV) python=$(PY_VERSION); \
	fi

ffmpeg: env ## Install ffmpeg for TorchCodec video decode (conda-forge)
	@echo "→ installing ffmpeg into '$(CONDA_ENV)' (needed by TorchCodec for dataset video)"
	@$(CONDA) install -y -n $(CONDA_ENV) -c conda-forge ffmpeg \
	  || echo "⚠ ffmpeg install failed; if you hit torchcodec/libsvtav1 errors try: $(CONDA) install -y -n $(CONDA_ENV) -c conda-forge ffmpeg=7.1.1"

clone: ## Clone LeRobot at the pinned commit
	@if [ ! -d "$(LEROBOT_DIR)/.git" ]; then \
	  echo "→ cloning $(LEROBOT_REPO)"; git clone $(LEROBOT_REPO) $(LEROBOT_DIR); \
	fi
	@echo "→ checking out $(LEROBOT_REF)"
	@git -C $(LEROBOT_DIR) fetch --quiet origin
	@git -C $(LEROBOT_DIR) checkout --quiet $(LEROBOT_REF)

install: env clone ## Install lerobot with configured extras into the full env (editable)
	@echo "→ installing lerobot[$(EXTRAS)] (first run downloads ~2.5 GB)…"
	$(ENV_PY) -m pip install -e "$(LEROBOT_DIR)[$(EXTRAS)]"

verify: ## Sanity-check the full env
	@$(ENV_PY) -c "import lerobot, torch, cv2, numpy, requests; print('lerobot :', lerobot.__file__); print('torch   :', torch.__version__, '| cuda:', torch.cuda.is_available()); print('cv2     :', cv2.__version__, '| numpy:', numpy.__version__, '| requests:', requests.__version__); print('OK ✅')"

verify-lite: ## Sanity-check the lite env
	@$(LITE_PY) -c "import cv2, numpy, requests; print('cv2', cv2.__version__, '| numpy', numpy.__version__, '| requests', requests.__version__, '| imshow OK'); print('OK ✅')"

info: ## Show env/tool paths and lerobot-info
	@echo "full env : $(CONDA_ENV)  ($(ENV_PY))"
	@echo "lite env : $(LITE_ENV)  ($(LITE_PY))"
	@echo "lerobot  : $(LEROBOT_DIR) @ $(LEROBOT_REF)"
	@conda run -n $(CONDA_ENV) lerobot-info 2>/dev/null || true

find-port: ## Identify the SO-101 serial ports (arm machine)
	conda run -n $(CONDA_ENV) lerobot-find-port

find-cameras: ## List detected cameras (arm machine)
	conda run -n $(CONDA_ENV) lerobot-find-cameras

bootstrap: ## Opt-in: install Miniforge to ~/miniforge3 (only if you have no conda)
	@echo "→ downloading Miniforge…"
	curl -fsSL -o /tmp/miniforge.sh \
	  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$$(uname)-$$(uname -m).sh
	bash /tmp/miniforge.sh -b -p $$HOME/miniforge3
	$$HOME/miniforge3/bin/conda init $$(basename $$SHELL)
	@echo "✅ Miniforge installed. Restart your shell, then run: make (or: make lite)"

apt-deps: ## (Linux, optional) system build deps — only if you hit build errors
	sudo apt-get install -y cmake build-essential python3-dev pkg-config \
	  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
	  libswscale-dev libswresample-dev libavfilter-dev

clean: ## Remove the LeRobot clone (keeps the conda envs)
	rm -rf $(LEROBOT_DIR)

clean-env: ## Delete both conda envs (destructive)
	-$(CONDA) env remove -y -n $(CONDA_ENV)
	-$(CONDA) env remove -y -n $(LITE_ENV)

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*## "}{printf "  make %-13s %s\n", $$1, $$2}'
