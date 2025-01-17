use half::f16;

use crate::{
    quantized::utils::{group_for_quantization, quantize_row_iq4_nl},
    Result,
};

use super::{BlockQ8K, GgmlDType, GgmlType, QK_K};

pub const QK4_NL: usize = 32;

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

const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

impl GgmlType for BlockIQ4xs {
    const DTYPE: GgmlDType = GgmlDType::IQ4_XS;
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
        if k % Self::BLCK_SIZE != 0 {
            crate::bail!("{k} is not divisible by {}", Self::BLCK_SIZE);
        };
        let nb = k / Self::BLCK_SIZE;
        if ys.len() != nb {
            crate::bail!(
                "size mismatch {} {} {}",
                xs.len(),
                ys.len(),
                Self::BLCK_SIZE
            )
        }
        const SUPER_BLOCK_SIZE: usize = QK_K;
        const BLOCK_SIZE: usize = 32;
        const NTRY: i32 = 7;

        let nrow = 1;
        let n_per_row = 256;
        let nblock = n_per_row / QK_K;

        for row in 0..nrow {
            let ys_block = &mut ys[nblock * row];
            for ibl in 0..nblock {
                let xs_block = &xs[n_per_row * row + QK_K * ibl..];
                quantize_row_iq4_nl(
                    xs_block,
                    SUPER_BLOCK_SIZE,
                    BLOCK_SIZE,
                    &mut ys_block.d,
                    &mut ys_block.qs,
                    &mut [ys_block.scales_h],
                    &mut ys_block.scales_l,
                    &KVALUES_IQ4NL,
                    None,
                    NTRY,
                );
            }
        }

        // for (ys_block, xs_block) in group_for_quantization(xs, ys)? {
        //     quantize_row_iq4_nl(
        //         xs_block,
        //         SUPER_BLOCK_SIZE,
        //         BLOCK_SIZE,
        //         &mut ys_block.d,
        //         &mut ys_block.qs,
        //         &mut [ys_block.scales_h],
        //         &mut ys_block.scales_l,
        //         &KVALUES_IQ4NL,
        //         None,
        //         NTRY,
        //     );
        // }
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
