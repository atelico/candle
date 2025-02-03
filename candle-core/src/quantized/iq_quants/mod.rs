use half::f16;
use utils::quantize_row_iq4_nl_impl;

use crate::{bail, Result};

mod utils;

use super::{BlockQ8K, GgmlDType, GgmlType, QK_K};

pub const QK4_NL: usize = 32;

const KVALUES_IQ4NL: [i8; 16] = [
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

    fn to_float(xs: &[Self], mut ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("dequantize block iq4xs {k} is not divisible by {QK_K}");
        }

        let nb = k / QK_K;
        for i in 0..nb {
            let block = &xs[i];

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
        todo!()
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }
}

fn quantize_iq4_xs(
    src: &[f32],
    ys: &mut [BlockIQ4xs],
    nrow: usize,
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Result<()> {
    // Basic sanity checks, similar to the C macro GGML_ASSERT
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

    // We'll need some local buffers that match the usage in the C code
    let mut lbuf = vec![0u8; QK_K]; // L[QK_K]
    let mut weight = vec![0f32; 32]; // weight[32] (the block_size is 32)
    let mut scales = vec![0f32; QK_K / 32]; // scales[QK_K/32], e.g. 256/32=8

    let mut src_offset = 0;
    let mut dst_offset = 0;

    for _row in 0..nrow {
        // Each row has `nblock` blocks:
        for ibl in 0..nblock {
            // In C:  block_iq4_xs * iq4 = (block_iq4_xs *)qrow;
            let block = &mut ys[dst_offset + ibl];

            // quant_weights?
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
        // In C: block_iq4_xs * iq4 = (block_iq4_xs *)qrow;
        // Here: let block_slice = &mut dst[dst_offset..dst_offset + nblock];
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
