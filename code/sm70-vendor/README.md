# Vendored third-party headers — SM70 D256 attention plugin

`code/ggml-cuda/fattn-sm70-d256.cu` (and the `-decode.cu` prototype) include headers from two
upstream projects. They are **not** shipped in this repository (13.8 MB, 160 files); fetch them
into `ggml/src/ggml-cuda/sm70-vendor/` before building:

## cute/ + cutlass/ (152 headers)

- Source: <https://github.com/NVIDIA/cutlass>
- Commit: `62750a2b75c802660e4894434dc55e839f322277`
- License: **BSD-3-Clause** (see `LICENSE-cute-cutlass` — *not* Apache-2.0)
- Why: the 1Cat SM70 D256 Split-D flash-attention kernel is written in CuTe
  (`MMA_Atom<SM70_8x8x4>` etc.); llama.cpp does not vendor CuTe.

## flash/ (8 headers)

- Source: <https://github.com/zhinianqin/flash-attention-v100>
- Commit: `c2eda5e6115b98c3ba4bfd181570668742eece22`
- License: **BSD-3-Clause** (see `LICENSE-flash-attention`)
- Why: online-softmax base layer, masking and kernel traits taken from the FA2 lineage fork the
  1Cat D256 patch builds on.

Suggested commands (run inside the llama.cpp checkout, after applying the patches):

```bash
git clone --filter=blob:none https://github.com/NVIDIA/cutlass /tmp/cutlass
git -C /tmp/cutlass checkout 62750a2b75c802660e4894434dc55e839f322277
mkdir -p ggml/src/ggml-cuda/sm70-vendor
cp -r /tmp/cutlass/include/cute   ggml/src/ggml-cuda/sm70-vendor/cute
cp -r /tmp/cutlass/include/cutlass ggml/src/ggml-cuda/sm70-vendor/cutlass

git clone https://github.com/zhinianqin/flash-attention-v100 /tmp/fav100
git -C /tmp/fav100 checkout c2eda5e6115b98c3ba4bfd181570668742eece22
cp -r /tmp/fav100/csrc/flash/* ggml/src/ggml-cuda/sm70-vendor/flash/
```

`CMakeLists.txt` only adds the include path and the CUDA 17/`M_LOG2E` flags when
`sm70-vendor/` exists; without it, delete `fattn-sm70-d256*.cu` (the prefill kernel is optional,
`LLAMA_SM70_D256=0` also disables it at runtime) or the build will fail on missing includes.
