# Vendored third-party headers — SM70 D256 attention plugin

`code/ggml-cuda/fattn-sm70-d256.cu` (and the `-decode.cu` prototype) need three header sets
under `ggml/src/ggml-cuda/sm70-vendor/`. Two of them come from upstream; the FA set is **shipped
here** because part of it is ours (see below).

## flash/ — shipped in this repository (7 headers)

```bash
cp -r code/sm70-vendor/flash  <llama.cpp>/ggml/src/ggml-cuda/sm70-vendor/
```

* 4 files are **upstream** from <https://github.com/zhinianqin/flash-attention-v100>
  @ `c2eda5e6115b98c3ba4bfd181570668742eece22` (BSD-3-Clause): `kernel_traits.h`,
  `namespace_config.h`, `philox.cuh`, `softmax.h`.
* 2 files carry **our port fixes** (marked inline with `local:` comments):
  * `mask.h` — `std::min/std::max` replaced with ternaries (no host functions in device code);
  * `utils.h` — the dead paged-KV helper `resolve_thread_kv_page_slice_offset` was dropped
    (it is the only user of `<optional>` in that header and is never instantiated here).
* 1 file is **ours**: `flash_namespace_config.h`. Upstream generates it in its CMake; this is the
  standalone stub that pins `FLASH_NAMESPACE` to `flash_sm70`.

The copied layout must be `sm70-vendor/flash/*.h` — the kernel includes them as
`#include "flash/mask.h"` etc. (the `sm70-vendor` directory is added to the include path by the
guarded block in `ggml/src/ggml-cuda/CMakeLists.txt`).

## cute/ + cutlass/ — fetch from upstream (152 headers)

```bash
git clone --filter=blob:none https://github.com/NVIDIA/cutlass /tmp/cutlass
git -C /tmp/cutlass checkout 62750a2b75c802660e4894434dc55e839f322277
mkdir -p <llama.cpp>/ggml/src/ggml-cuda/sm70-vendor
cp -r /tmp/cutlass/include/cute    <llama.cpp>/ggml/src/ggml-cuda/sm70-vendor/cute
cp -r /tmp/cutlass/include/cutlass <llama.cpp>/ggml/src/ggml-cuda/sm70-vendor/cutlass
```

* Source: <https://github.com/NVIDIA/cutlass>, commit `62750a2b75c802660e4894434dc55e839f322277`
* License: **BSD-3-Clause** (see `LICENSE-cute-cutlass` — *not* Apache-2.0)
* Why: the 1Cat SM70 D256 Split-D kernel is written in CuTe (`MMA_Atom<SM70_8x8x4>` etc.);
  llama.cpp does not vendor CuTe. A full copy of `include/{cute,cutlass}` is fine; llama.cpp's
  build only compiles the plugin and its transitive includes.

## If you do not want the D256 kernel

Delete `fattn-sm70-d256*.cu` after applying the patches (or set `LLAMA_SM70_D256=0` at runtime);
`CMakeLists.txt` only adds the extra include path and the CUDA-17/`M_LOG2E` flags when
`sm70-vendor/` exists, but the `.cu` files are picked up by a wildcard, so without the headers the
build fails on missing includes.

Both license texts that must travel with this directory are in `code/sm70-vendor/`
(`LICENSE-cute-cutlass`, `LICENSE-flash-attention`).
