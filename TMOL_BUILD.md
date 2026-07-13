# Building & Using tmol on Wynton

[`tmol`](https://github.com/uw-ipd/tmol) (UW-IPD) is a GPU-accelerated PyTorch
reimplementation of Rosetta's `beta_nov2016_cart` energy function with custom
C++/CUDA kernels. We use it as a reward (energy score) for guided sampling — it's
strategy-agnostic, usable with FK(C) steering, pure guidance, DPS, etc. This note
records how it's wired into the pixi environments and how to build/run it on
Wynton, because the install is non-trivial.

## TL;DR

- tmol is a pypi-dependency of the `tmol` pixi **feature**, attached to every
  model env (`boltz*`, `protenix*`, `rf3*`). See `[tool.pixi.feature.tmol*]` in
  `pyproject.toml`.
- It is **compiled from source** (no usable prebuilt wheel — see below), so a
  clean build needs a CUDA toolchain (present in the linux-64 envs via
  `cuda-toolkit`) and the cmake/parallelism settings baked into
  `[tool.pixi.feature.tmol.activation.env]`.
- Build it on a machine with `nvcc` (e.g. **gpudev1** — a GPU is *not* required
  to compile, only to run):

  ```bash
  cd /wynton/home/fraserlab/<you>/samplew0rks
  PIXI_CACHE_DIR=/scratch/<you>/pixi-cache pixi install -e boltz-dev
  ```

- Run on an **sm_80 / sm_86** GPU node (that's what we compile kernels for):

  ```bash
  qrsh -q gpu.q -l compute_cap=80 -pe smp 8
  cd /wynton/home/fraserlab/<you>/samplew0rks
  pixi run -e boltz-dev python -c "import tmol, torch; print(tmol.__version__, torch.cuda.get_device_name())"
  ```

## Why source-built (no wheel)

tmol publishes an **sdist only** on PyPI; prebuilt wheels live on GitHub Releases
and are lane-specific (`python × torch-minor × CUDA`). The published GPU lanes
cover torch **2.8–2.12** only. This project is pinned to **torch 2.7.1**
(protenix hard-pins `torch==2.7.1`), so **no prebuilt wheel matches** and tmol
must compile from source. If protenix ever relaxes its torch pin and the project
moves to torch ≥ 2.8, tmol can switch to the fast prebuilt-wheel install and all
the cmake workarounds below can be deleted.

## The build workarounds (why the config looks the way it does)

All of these live in `[tool.pixi.feature.tmol.activation.env]` /
`[tool.pixi.pypi-options]` in `pyproject.toml`. They exist because tmol compiles
torch C++/CUDA extensions against a **conda** CUDA toolkit, which is laid out
differently than a system CUDA install.

1. **Build isolation off** (`[tool.pixi.pypi-options] no-build-isolation`).
   tmol needs the env's torch present at build time; an isolated build can't see
   it. Its PEP517 build deps (`scikit-build-core`, `pybind11`, `packaging`) are
   added to the feature so they're available without isolation. `ninja` comes
   from `[tool.pixi.dependencies]`.

2. **CUDA root** (`CUDA_TOOLKIT_ROOT_DIR` / `CUDAToolkit_ROOT` →
   `$CONDA_PREFIX/targets/x86_64-linux`). torch's legacy `FindCUDA` looks for
   headers in `<root>/include`, but conda puts them under
   `targets/x86_64-linux/include`. Without this it reports
   *"cannot find the CUDA libraries."*

3. **NVTX3 headers** (`USE_SYSTEM_NVTX=ON` + `CMAKE_INCLUDE_PATH` →
   `.../site-packages/nvidia/nvtx/include`). CUDA 12 dropped the old
   `nvToolsExt` library; torch needs header-only NVTX3, which conda's `cuda-nvtx`
   does **not** ship — only torch's bundled `nvidia-nvtx` wheel has the headers.
   Without this: *"Could NOT find nvtx3"* → dead `CUDA::nvToolsExt` target.

4. **Skip the wheel probe** (`TMOL_DISABLE_WHEEL_FETCH=1`). tmol's sdist backend
   otherwise probes GitHub Releases for a matching wheel (none exists for torch
   2.7) before falling back to a source build.

5. **CUDA architectures** (`CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;86"`).
   - Passed via **`CMAKE_ARGS`, not `SKBUILD_CMAKE_DEFINE`**: the latter splits
     entries on `;`, so a `;`-separated arch list crashes its parser
     (`not enough values to unpack ... Field cmake.define`). `CMAKE_ARGS` is
     shlex-split on spaces, so the `;` survives.
   - Targets **sm_80** (A100, 3 nodes) + **sm_86** (RTX 3090 / A40, 29 nodes) —
     everything a `-l compute_cap=80` request can land on.

6. **Bounded parallelism** (`CMAKE_BUILD_PARALLEL_LEVEL=6`). tmol's CUDA kernels
   are template-heavy: each `nvcc` front-end (`cicc`) peaks ~3.4 GB, and it
   builds **all requested archs concurrently per file**. At ninja's default
   (`cores+2`) this OOM-kills the build against **Wynton's 48 GB/user cgroup**.
   At 2 archs, `-j6` peaks ~24–34 GB (measured 24 GB). Must live in
   `activation.env` — **pixi does not forward shell `export`s into the build
   subprocess** (only vars declared in `activation.env` reach it).

## Gotchas

- **`PIXI_CACHE_DIR` must be on local disk (`/scratch`).** The default cache on
  the NFS home fails uv's atomic renames mid-build
  (`Resource busy (os error 16)`). Not committed to `pyproject.toml` because it's
  per-machine; set it on the command line or in your shell profile.
- **uv caches the built wheel ignoring these settings.** If you change the arch
  list (or any cmake setting) and rebuild, uv may silently reuse the old wheel
  and exit 0 without recompiling. Force a real rebuild by purging the cache and
  the installed copy:
  ```bash
  C=/scratch/<you>/pixi-cache/uv-cache
  find $C/sdists-v9/pypi/tmol -iname 'tmol*.whl' -printf '%h\n' | sort -u | xargs -r rm -rf
  find $C/archive-v0 -maxdepth 2 -type d -iname 'tmol*' | xargs -r -n1 dirname | sort -u | xargs -r rm -rf
  rm -rf .pixi/envs/boltz-dev/lib/python3.12/site-packages/tmol*
  ```
- **Verify the binary, not the exit code.** Confirm which archs actually got
  compiled in:
  ```bash
  cuobjdump --list-elf .pixi/envs/boltz-dev/lib/python3.12/site-packages/tmol/_C*.so | grep -oE 'sm_[0-9]+' | sort -u
  ```
- **gpudev1 can build but not run tmol.** Its GPUs are GTX 1080 (sm_61), which
  isn't in our arch list — `import tmol` works there, but CUDA ops won't.

## Adding more GPU architectures

To also run on e.g. L40/RTX 4090 (sm_89) or H100 (sm_90), extend the arch list
in `CMAKE_ARGS` (`-DCMAKE_CUDA_ARCHITECTURES=80;86;89;90`). Each extra arch adds
a concurrent `cicc` (~3.4 GB) per file, so **lower `CMAKE_BUILD_PARALLEL_LEVEL`**
to stay under the 48 GB cap (roughly: `48 GB / (archs × 4 GB)` jobs). Then purge
the uv cache (above) so the rebuild actually happens.

## Wynton GPU fleet (compute capability → nodes)

| compute_cap | Arch / cards | # nodes | tmol runs? |
|---|---|---|---|
| 86 | Ampere (RTX 3090 / A40) | 29 | ✅ (built) |
| 80 | Ampere (A100) | 3 | ✅ (built) |
| 75 | Turing (2080 Ti / T4) | 14 | ❌ |
| 61 | Pascal (GTX 1080, gpudev1) | 18 | ❌ |
| ~90 | Hopper (H100) | 1 | ❌ (add sm_90) |

Request a node with `qrsh -q gpu.q -l compute_cap=80 -pe smp <N>` (matches
nodes with capability ≥ 8.0; the job gets the node exclusively).
