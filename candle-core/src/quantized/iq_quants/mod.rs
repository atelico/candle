use half::f16;
use utils::quantize_row_iq4_nl_impl;

use crate::{bail, Result};

mod utils;

use super::{BlockQ8K, GgmlDType, GgmlType, QK_K};

pub const QK4_NL: usize = 32;

pub(super) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ4xs {
    pub(crate) d: f16,
    pub(crate) scales_h: u16,
    pub(crate) scales_l: [u8; QK_K / 64],
    pub(crate) qs: [u8; QK_K / 2],
}

const _: () = assert!(
    std::mem::size_of::<BlockIQ4xs>()
        == std::mem::size_of::<f16>() + std::mem::size_of::<u16>() + QK_K / 64 + QK_K / 2,
    "wrong iq4_xs block size/padding"
);

impl GgmlType for BlockIQ4xs {
    const DTYPE: GgmlDType = GgmlDType::Iq4Xs;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;
    const SUPPORTS_I8MM: bool = false;

    fn to_float(xs: &[Self], mut ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("dequantize block iq4xs {k} is not divisible by {QK_K}");
        }

        let nb = k / QK_K;
        for block in xs.iter().take(nb) {
            let d = block.d.to_f32();
            let mut qs = &block.qs[..];

            for ib in 0..(QK_K / 32) {
                let ib_div_2 = ib / 2;
                let ib_mod_2 = ib % 2;

                let ls_low = (block.scales_l[ib_div_2] as i32 >> (4 * ib_mod_2 as i32)) & 0xF;
                let ls_high = ((block.scales_h as i32 >> (2 * ib as i32)) & 3) << 4;
                let ls = ls_low | ls_high;

                let dl = d * (ls as f32 - 32.);

                for j in 0..16 {
                    ys[j] = dl * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
                    ys[j + 16] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }

                qs = &qs[16..];
                ys = &mut ys[32..];
            }
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        let k = xs.len();
        if k % QK_K != 0 {
            bail!("Input length must be multiple of QK_K = {}", QK_K);
        }

        println!("starting");
        quantize_iq4_xs(xs, ys, 1, k, None)?;

        Ok(())
    }

    fn from_float_imatrix(
        xs: &[f32],
        ys: &mut [Self],
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let k = xs.len();
        if k % QK_K != 0 {
            bail!("Input length must be multiple of QK_K = {}", QK_K);
        }
        let nrow = xs.len() / n_per_row;

        quantize_iq4_xs_imatrix(xs, ys, nrow, n_per_row, Some(imatrix_weights));

        Ok(())
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        // #[cfg(target_feature = "avx")]
        // todo!();

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_iq4_xs_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if n % QK_K != 0 {
            bail!("n must be a multiple of QK_K");
        }
        let nb = n / QK_K;

        let mut sumf = 0.0f32;

        // Loop over each block
        for ibl in 0..nb {
            // x[ibl], y[ibl]
            let x = &xs[ibl];
            let y = &ys[ibl];

            // Convert x.d from fp16 to fp32, then multiply by y.d
            let d4d8 = x.d.to_f32() * y.d;

            // We'll track the "h" scales
            let mut h = x.scales_h;

            let mut qsi = 0; // index for x.qs
            let mut q8i = 0; // index for y.qs

            // so we step by 2 in Rust as well
            for ib2 in (0..(QK_K / 32)).step_by(2) {
                // Reproduce the logic of ls1, ls2
                let ls1 = (x.scales_l[ib2 / 2] & 0x0f) | (((h << 4) & 0x30) as u8);
                let ls2 = (x.scales_l[ib2 / 2] >> 4) | (((h << 2) & 0x30) as u8);
                // Then we shift h by 4 in the original code
                h >>= 4;

                // Convert ls1, ls2 to "scaled" floats
                let d1 = d4d8 * ((ls1 as i32) - 32) as f32;
                let d2 = d4d8 * ((ls2 as i32) - 32) as f32;

                // Two sets of 16 items each
                // sum of the first 16-lane block
                let mut sumi1 = 0;
                let mut sumi2 = 0;

                // The first pass
                for j in 0..16 {
                    // q8i + j   vs q8i + j + 16
                    // qs[qsi + j] & 0xf vs (qs[qsi + j] >> 4)
                    sumi1 += (y.qs[q8i + j] as i32)
                        * KVALUES_IQ4NL[(x.qs[qsi + j] & 0x0f) as usize] as i32;
                    sumi2 += (y.qs[q8i + j + 16] as i32)
                        * KVALUES_IQ4NL[((x.qs[qsi + j] >> 4) & 0x0f) as usize] as i32;
                }
                sumf += d1 * ((sumi1 + sumi2) as f32);

                qsi += 16;
                q8i += 32;

                // The second pass
                sumi1 = 0;
                sumi2 = 0;
                for j in 0..16 {
                    sumi1 += (y.qs[q8i + j] as i32)
                        * KVALUES_IQ4NL[(x.qs[qsi + j] & 0x0f) as usize] as i32;
                    sumi2 += (y.qs[q8i + j + 16] as i32)
                        * KVALUES_IQ4NL[((x.qs[qsi + j] >> 4) & 0x0f) as usize] as i32;
                }
                sumf += d2 * ((sumi1 + sumi2) as f32);

                qsi += 16;
                q8i += 32;
            }
        }

        Ok(sumf)
    }

    #[allow(unused)]
    #[cfg(feature = "arm-nightly-feat")]
    fn matmul_i8mm(
        n: usize,
        xs_0: &[Self],
        xs_1: &[Self],
        ys_0: &[Self::VecDotType],
        ys_1: &[Self::VecDotType],
    ) -> Result<[f32; 4]> {
        crate::bail!("Unsupported block type for i8mm");
    }
}

fn quantize_iq4_xs(
    src: &[f32],
    ys: &mut [BlockIQ4xs],
    nrow: usize,
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Result<()> {
    if n_per_row % QK_K != 0 {
        bail!("n_per_row must be multiple of QK_K = {}", QK_K);
    }

    let nblock = n_per_row / QK_K;
    // We expect exactly nrow * nblock blocks in `ys`.
    if ys.len() != nrow * nblock {
        bail!(
            "Output buffer size mismatch: want {} blocks, got {}",
            nrow * nblock,
            ys.len()
        );
    }

    let mut lbuf = vec![0u8; QK_K]; // L[QK_K]
    let mut weight = vec![0f32; 32]; // weight[32] (the block_size is 32)
    let mut scales = vec![0f32; QK_K / 32]; // scales[QK_K/32], e.g. 256/32=8

    let mut src_offset = 0;
    let mut dst_offset = 0;

    for _row in 0..nrow {
        // Each row has `nblock` blocks:
        for ibl in 0..nblock {
            println!("ibl {ibl}");
            let block = &mut ys[dst_offset + ibl];

            let qw = quant_weights.map(|qw_all| {
                let start = QK_K * ibl;
                &qw_all[start..start + QK_K]
            });

            quantize_row_iq4_nl_impl(
                /* super_block_size = */ QK_K,
                /* block_size       = */ 32,
                /* x                = */
                &src[src_offset + QK_K * ibl..src_offset + QK_K * (ibl + 1)],
                /* dh               = */ &mut block.d,
                /* q4               = */ &mut block.qs,
                /* scales_h         = */ &mut block.scales_h,
                /* scales_l         = */ &mut block.scales_l,
                /* scales           = */ &mut scales,
                /* weight           = */ &mut weight,
                /* L                = */ &mut lbuf,
                /* values           = */ &KVALUES_IQ4NL,
                /* quant_weights    = */ qw,
                /* ntry             = */ 7,
            );
        }
        src_offset += n_per_row;
        dst_offset += nblock;
    }

    Ok(())
}

pub fn quantize_iq4_xs_imatrix(
    src: &[f32],
    dst: &mut [BlockIQ4xs],
    nrow: usize,
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) {
    // 1. Check that n_per_row is multiple of QK_K
    assert_eq!(n_per_row % QK_K, 0, "n_per_row must be multiple of QK_K");
    let nblock = n_per_row / QK_K;

    // 2. We expect nrow * nblock blocks in `dst`
    assert_eq!(
        dst.len(),
        nrow * nblock,
        "Output slice must have exactly nrow*nblock elements"
    );

    // 3. Local buffers matching the C usage
    let mut lbuf = vec![0u8; QK_K];
    let mut weight = vec![0f32; 32];
    let mut scales = vec![0f32; QK_K / 32];

    // We'll track how far we've consumed `src`.
    let mut src_offset = 0;
    // Also track how far we move in `dst`.
    let mut dst_offset = 0;

    // 4. Outer loop over rows
    for _row in 0..nrow {
        for ibl in 0..nblock {
            let block = &mut dst[dst_offset + ibl];

            // If quant_weights is Some, get the sub-slice for this block
            let qw_block = quant_weights.map(|qw_all| &qw_all[ibl * QK_K..(ibl + 1) * QK_K]);

            quantize_row_iq4_nl_impl(
                QK_K, // super_block_size
                32,   // block_size
                &src[src_offset + ibl * QK_K..src_offset + (ibl + 1) * QK_K],
                &mut block.d,
                &mut block.qs,
                &mut block.scales_h,
                &mut block.scales_l,
                &mut scales,
                &mut weight,
                &mut lbuf,
                &KVALUES_IQ4NL,
                qw_block,
                7, // ntry
            );
        }
        src_offset += n_per_row;
        dst_offset += nblock;
    }
}
