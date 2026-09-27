// sm_70 decode-time flash attention for D=256 with q8_0 K/V.
//
// Why a new kernel: on Volta our production model (head_dim 256, 4 kv heads, gqa 6, q8_0 KV)
// cannot use any of the stock kernels well at decode time.
//   * flash_attn_ext_vec: used at q_len == 1, but it re-reads K/V once per q head (6x per kv head)
//     and once per q tile, and it cannot be used at all for q_len >= 2 without crashing.
//   * flash_attn_tile: used for q_len >= 2; measured 4% DRAM utilisation, 12% warp occupancy,
//     8x L2 amplification over DRAM (see RESEARCH-V100 18.7/18.8).
//   * flash_attn_ext_f16 (MMA): would materialise a session-sized f16 copy of K/V per op.
//
// This kernel reads K/V exactly once per (kv head, q tile): a block covers all `ncols2` q heads
// that share one kv head and `ncols1` (<= 4) query rows, and it stages K/V tiles in shared memory
// as raw q8_0 (16-byte loads), computing Q.K with dp4a against a q8-quantised Q. The softmax is
// online per (row, head) pair, and the V accumulation keeps 8 fp32 dims per lane.

#include "common.cuh"
#include "fattn-common.cuh"

#define SM70_D256_DEC_NTHREADS 256
#define SM70_D256_DEC_TILE_KV  32   // KV tokens staged in shared memory per iteration
// One (row, head) pair = 256 values = 8 blocks of 32 q8_0 values.
static constexpr __device__ int sm70_d256_dec_nblk() { return 8; }

// Warp-reduce helper that works on the 32 lanes of a warp.
template <int width>
static __device__ __forceinline__ float sm70_dec_warp_sum(float v) {
    static_assert(width == 32, "only full warps supported");
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, o, 32);
    }
    return v;
}

template <int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(SM70_D256_DEC_NTHREADS, 2)
static __global__ void flash_attn_ext_decode_q8_sm70(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    GGML_UNUSED(ne00); GGML_UNUSED(ne13); GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(ne33);
    GGML_UNUSED(nb32); GGML_UNUSED(nb33); GGML_UNUSED(sinks_ptr);

    constexpr int D         = 256;
    constexpr int NBLK      = D/32;            // q8_0 blocks per row
    constexpr int KV_ROW    = NBLK*sizeof(block_q8_0);   // 272 bytes per KV row
    constexpr int NP        = ncols1*ncols2;   // (row, head) pairs per block
    constexpr int LANE_DIMS = D/32;            // 8 dims per lane
    constexpr int TILE      = SM70_D256_DEC_TILE_KV;

    const int nthreads = SM70_D256_DEC_NTHREADS;
    const int tid      = threadIdx.x + threadIdx.y*32;
    const int warp     = threadIdx.y;
    const int lane     = threadIdx.x;
    constexpr int nwarps = SM70_D256_DEC_NTHREADS/32;

    // ---- which (sequence, kv head, q heads, rows) does this block own? ------------------------
    const int q_len      = int(ne01.z);
    const int gqa_ratio  = ne02 / ne12;
    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1)/ncols2;
    const int sequence   = blockIdx.z / (ntiles_z_gqa*ne12);
    const int zrem       = blockIdx.z % (ntiles_z_gqa*ne12);
    const int kv_head    = zrem / ntiles_z_gqa;
    const int gqa_tile   = zrem % ntiles_z_gqa;
    const int head0      = kv_head*gqa_ratio + gqa_tile*ncols2;
    const int col0       = blockIdx.x*ncols1;

    const char * Q = Q_ptr + int64_t(nb03)*sequence;
    const char * K = K_ptr + int64_t(nb13)*sequence + int64_t(nb12)*kv_head;
    const char * V = V_ptr + int64_t(nb23)*sequence + int64_t(nb22)*kv_head;

    const half * maskh = mask_ptr ? (const half *)(mask_ptr + nb31*col0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head0, n_head_log2, m0, m1);

    const int kv_end = KV_max_ptr ? KV_max_ptr[sequence*gridDim.x + blockIdx.x] : ne11;
    const int kv_per_block = gridDim.y;

    // ---- shared memory -----------------------------------------------------------------------
    // K/V tiles are staged as separate planes: 256 raw qs bytes per row (4-byte aligned so the
    // inner loop can use int loads) plus the 8 per-block scales. Staging the raw 34-byte blocks
    // instead would put every qs at a 2-byte-only offset, and int loads there trap with
    // "misaligned address".
    __shared__ __align__(16) int8_t sKq[TILE][256];
    __shared__ __align__(16) half   sVh[TILE][256];   // V dequantised once per tile (fp16 storage,
                                                      // fp32 accumulate): saves 8 int8->float
                                                      // conversions per lane per (token, pair)
    __shared__ __align__(16) half   sKd[TILE][8];
    __shared__ float                sVd[TILE][8];
    __shared__ __align__(16) int   sQ[NP][NBLK*8];      // quantised Q, 32 int8 per block
    __shared__ __align__(16) float sQd[NP][NBLK];       // per block scale of Q
    __shared__ float sScore[NP][TILE];                  // per-tile softmax weights, lane == token

    // ---- quantise Q to q8 (per 32-value block) -----------------------------------------------
    // NP*NBLK blocks of 32 values; one thread per block.
    for (int task = tid; task < NP*NBLK; task += nthreads) {
        const int p = task / NBLK;
        const int b = task % NBLK;
        const int col = col0 + p / ncols2;
        const int hd  = head0 + p % ncols2;
        int * qs = sQ[p] + b*8;
        float d = 0.0f;
        if (col < q_len) {
            const float * qf = (const float *)(Q + int64_t(nb01)*col + int64_t(nb02)*hd);
            const float * xs = qf + b*32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = fmaxf(amax, fabsf(xs[i]));
            }
            d = amax > 0.0f ? amax/127.0f : 0.0f;
            const float id = d > 0.0f ? 1.0f/d : 0.0f;
            for (int i = 0; i < 32; i += 4) {
                // mask each byte: a negative quant would otherwise shift its sign bits into the
                // neighbouring bytes (e.g. -5 << 8 == 0xFFFFFB00) and corrupt them.
                const int v = (((int) roundf(xs[i + 0]*id)) & 0xFF)
                            | ((((int) roundf(xs[i + 1]*id)) & 0xFF) <<  8)
                            | ((((int) roundf(xs[i + 2]*id)) & 0xFF) << 16)
                            | ((((int) roundf(xs[i + 3]*id)) & 0xFF) << 24);
                qs[i/4] = v;
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                qs[i] = 0;
            }
        }
        sQd[p][b] = d*scale;
    }
    __syncthreads();

    // ---- KV loop ------------------------------------------------------------------------------
    // Each warp handles the pairs warp, warp+nwarps, ... Each pair keeps its own online softmax
    // state (m, l) and an 8-float accumulator over this lane's dims.
    float acc[ncols1*ncols2/nwarps + 1][LANE_DIMS];
    float pmax[ncols1*ncols2/nwarps + 1];
    float psum[ncols1*ncols2/nwarps + 1];
    for (int s = 0; s < ncols1*ncols2/nwarps + 1; ++s) {
        pmax[s] = -FLT_MAX/2.0f;
        psum[s] = 0.0f;
        for (int i = 0; i < LANE_DIMS; ++i) {
            acc[s][i] = 0.0f;
        }
    }

    const int kv_begin = blockIdx.y*TILE;
    const int kv_step  = kv_per_block*TILE;

    for (int k0 = kv_begin; k0 < kv_end; k0 += kv_step) {
        const int tile_n = min(TILE, kv_end - k0);

        // stage K/V tile (bounds-checked, zero-filled beyond tile_n)
        // one task = one (token, q8_0 block) pair of K, and the same for V
        for (int task = tid; task < tile_n*NBLK; task += nthreads) {
            const int t = task / NBLK;
            const int b = task % NBLK;
            const char * kblk = K + int64_t(k0 + t)*nb11 + b*sizeof(block_q8_0);
            const char * vblk = V + int64_t(k0 + t)*nb21 + b*sizeof(block_q8_0);
            *(half *) &sKd[t][b] = *((const half *) kblk);
            sVd[t][b]             = __half2float(*((const half *) vblk));
            ggml_cuda_memcpy_1<16, 2>(&sKq[t][b*32],      kblk + 2);
            ggml_cuda_memcpy_1<16, 2>(&sKq[t][b*32 + 16], kblk + 18);
            // dequantise V: 32 int8 * scale -> 16 half2
            {
                const int8_t * v8 = (const int8_t *)(vblk + 2);
                const float    dv = sVd[t][b];
                half2 * dst = (half2 *) &sVh[t][b*32];
                for (int i = 0; i < 16; ++i) {
                    dst[i] = __floats2half2_rn(dv*(float)v8[2*i + 0], dv*(float)v8[2*i + 1]);
                }
            }
        }
        __syncthreads();

        // Two passes per tile (the shape the tile kernel uses): in pass A every lane computes the
        // full 256-dim dot of its own token, so there is no per-token warp reduction -- the only
        // dependency chain left is the lane's own 64 dp4a. Pass B then lets every lane accumulate
        // its own 8 dimensions across all tokens of the tile.
        for (int s = 0, p = warp; p < NP; p += nwarps, ++s) {
            // ---- pass A: lane == token
            float score = -FLT_MAX/2.0f;
            if (lane < tile_n) {
                const int * q32 = sQ[p];
                const int * k32 = (const int *) sKq[lane];
                float s_dot = 0.0f;
#pragma unroll
                for (int b = 0; b < NBLK; ++b) {
                    int dot = 0;
#pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        dot = __dp4a(k32[b*8 + i + 0], q32[b*8 + i + 0], dot);
                        dot = __dp4a(k32[b*8 + i + 1], q32[b*8 + i + 1], dot);
                    }
                    s_dot += (float) dot * sQd[p][b] * __half2float(sKd[lane][b]);
                }
                score = s_dot;
                if (use_logit_softcap) {
                    score = logit_softcap*tanhf(score);
                }
                if (maskh != nullptr) {
                    score += slope*__half2float(maskh[(p/ncols2)*ne11 + k0 + lane]);
                }
            }

            // one max reduction per tile (not per token), then the tile's weights
            float m_tile = score;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFFu, m_tile, o, 32));
            }
            const float m_new = fmaxf(pmax[s], m_tile);
            const float alpha = __expf(pmax[s] - m_new);
            const float p_t   = lane < tile_n ? __expf(score - m_new) : 0.0f;
            sScore[p][lane] = p_t;

            float tile_sum = p_t;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                tile_sum += __shfl_xor_sync(0xFFFFFFFFu, tile_sum, o, 32);
            }
            pmax[s] = m_new;
            psum[s] = psum[s]*alpha + tile_sum;
#pragma unroll
            for (int i = 0; i < LANE_DIMS; ++i) {
                acc[s][i] *= alpha;
            }

            // ---- pass B: lane owns dimensions [lane*8, lane*8+8)
            __syncwarp();
            const int     b    = lane/4;
            const half2 * v0   = (const half2 *)(sVh[0] + b*32 + (lane%4)*8);
            for (int t = 0; t < tile_n; ++t) {
                const float   vscale = sScore[p][t];
                const half2 * vh2    = v0 + t*128;
                const float2  a0 = __half22float2(vh2[0]);
                const float2  a1 = __half22float2(vh2[1]);
                const float2  a2 = __half22float2(vh2[2]);
                const float2  a3 = __half22float2(vh2[3]);
                acc[s][0] += vscale*a0.x; acc[s][1] += vscale*a0.y;
                acc[s][2] += vscale*a1.x; acc[s][3] += vscale*a1.y;
                acc[s][4] += vscale*a2.x; acc[s][5] += vscale*a2.y;
                acc[s][6] += vscale*a3.x; acc[s][7] += vscale*a3.y;
            }
        }
        __syncthreads();
    }

    // ---- write out ---------------------------------------------------------------------------
    // Parts layout (matches flash_attn_combine_results):
    //   parts[((seq*q_len + row)*ne02 + head)*gridDim.y*D + blockIdx.y*D + dim]
    //   meta [((seq*q_len + row)*ne02 + head)*gridDim.y + blockIdx.y] = (max, sum)
    for (int s = 0, p = warp; p < NP; p += nwarps, ++s) {
        const int col  = col0 + p/ncols2;
        const int head = head0 + p%ncols2;
        const int j_dst = (sequence*q_len + col)*ne02 + head;
        float * out = dst_ptr + (int64_t(j_dst)*gridDim.y + blockIdx.y)*D + lane*LANE_DIMS;
        if (gridDim.y == 1) {
            const float inv = 1.0f/psum[s];
#pragma unroll
            for (int i = 0; i < LANE_DIMS; ++i) {
                out[i] = acc[s][i]*inv;
            }
        } else {
#pragma unroll
            for (int i = 0; i < LANE_DIMS; ++i) {
                out[i] = acc[s][i];
            }
            if (lane == 0) {
                dst_meta_ptr[int64_t(j_dst)*gridDim.y + blockIdx.y] = make_float2(pmax[s], psum[s]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------
static bool ggml_cuda_sm70_d256_decode_geometry_ok(const int cc, const ggml_tensor * dst) {
    if (cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q == nullptr || K == nullptr || V == nullptr) return false;
    if (sinks != nullptr) return false;
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0) return false;
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) return false;
    // Only single-row decode for now: at nb >= 2 the stock tile kernel parallelises rows across
    // blocks and still wins (measured: kv=16384 nb=3 276 us vs 443 us here), while at nb == 1 this
    // kernel beats the tile kernel by 8-21% (kv=16384: 151 us vs 190 us). See RESEARCH 21.
    if (Q->ne[1] != 1) return false;
    if (Q->ne[3] != 1 || K->ne[3] != 1) return false;           // single sequence (v1)
    if (K->ne[1] < 1 || V->ne[1] != K->ne[1]) return false;
    if (K->ne[2] < 1 || V->ne[2] != K->ne[2]) return false;
    if (Q->ne[2] % K->ne[2] != 0) return false;
    if (Q->ne[2]/K->ne[2] != 6) return false;                   // only the compiled ncols2 == 6
    if (K->nb[0] != (int64_t) ggml_type_size(K->type)) return false;
    if (V->nb[0] != (int64_t) ggml_type_size(V->type)) return false;
    if (K->nb[1] < 8*(int64_t) sizeof(block_q8_0)) return false;
    if (V->nb[1] < 8*(int64_t) sizeof(block_q8_0)) return false;
    if (mask != nullptr) {
        if (mask->type != GGML_TYPE_F16) return false;
        if (mask->ne[1] < Q->ne[1]) return false;
        if (mask->nb[1] != (int64_t) K->ne[1]*2) return false;  // rows contiguous
    }
    if (ggml_cuda_batch_invariant()) return false;
    return true;
}

template <int ncols1, bool use_logit_softcap>
static void ggml_cuda_sm70_d256_decode_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps        = SM70_D256_DEC_NTHREADS/32;
    constexpr size_t nbytes_shared = 0;   // all shared memory is static
    fattn_kernel_t fattn_kernel = flash_attn_ext_decode_q8_sm70<ncols1, 6, use_logit_softcap>;
    launch_fattn<256, ncols1, 6>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, SM70_D256_DEC_TILE_KV,
                                 /*need_f16_K =*/ false, /*need_f16_V =*/ false, /*stream_k =*/ false);
}

void ggml_cuda_flash_attn_ext_sm70_d256_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // Rows per block: all rows of the batch share one K/V read, but a block then has to iterate
    // ncols1*6 pairs sequentially per warp. For the 3-row MTP shape the extra parallelism of
    // splitting rows across blocks measured better (GGML_SM70_D256_DEC_ROWS to A/B).
    const int ncols1 = (int) Q->ne[1];   // == 1 (see the gate); kept general for future tuning

    if (logit_softcap != 0.0f) {
        switch (ncols1) {
            case 1: ggml_cuda_sm70_d256_decode_impl<1, true >(ctx, dst); break;
            case 2: ggml_cuda_sm70_d256_decode_impl<2, true >(ctx, dst); break;
            case 3: ggml_cuda_sm70_d256_decode_impl<3, true >(ctx, dst); break;
            case 4: ggml_cuda_sm70_d256_decode_impl<4, true >(ctx, dst); break;
            default: GGML_ABORT("fatal error");
        }
        return;
    }
    switch (ncols1) {
        case 1: ggml_cuda_sm70_d256_decode_impl<1, false>(ctx, dst); break;
        case 2: ggml_cuda_sm70_d256_decode_impl<2, false>(ctx, dst); break;
        case 3: ggml_cuda_sm70_d256_decode_impl<3, false>(ctx, dst); break;
        case 4: ggml_cuda_sm70_d256_decode_impl<4, false>(ctx, dst); break;
        default: GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_sm70_d256_decode_supported(int cc, const ggml_tensor * dst) {
    const bool ok = ggml_cuda_sm70_d256_decode_geometry_ok(cc, dst);
    static const bool verbose = getenv("GGML_SM70_D256_DEC_DEBUG") != nullptr;
    if (verbose) {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const ggml_tensor * mask = dst->src[3];
        fprintf(stderr, "[sm70-dec] %s Q=(%lld,%lld,%lld,%lld) K=(%lld,%lld,%lld,%lld) type=%d/%d mask=%s mne=(%lld,%lld) mnb1=%lld\n",
                ok ? "ACCEPT" : "reject",
                (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                (int) K->type, (int) V->type,
                mask ? "yes" : "no",
                mask ? (long long) mask->ne[0] : -1, mask ? (long long) mask->ne[1] : -1,
                mask ? (long long) mask->nb[1] : -1);
    }
    return ok;
}
