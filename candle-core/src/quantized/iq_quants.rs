use half::f16;

use crate::{
    quantized::utils::{group_for_quantization, quantize_iq4_nl},
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

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("dequantis block iq4xs {k} is not divisible by {QK_K}");
        }

        let nb = k / QK_K;
        for i in 0..nb {
            let block = &xs[i];

            let d = block.d.to_f32();
            let qs = &block.qs;

            let mut qs_offset = 0;

            // A pointer (offset) into out_chunk:
            let mut y_offset = 0;

            // 2. For each sub-block of size 32:
            //    QK_K/32 sub-blocks, each sub-block contributes 32 floats of output.
            for ib in 0..(QK_K / 32) {
                // 2a. Reconstruct `ls` from scales_l/scales_h:
                //    This matches the C code:
                //    ls = ((scales_l[ib/2] >> (4*(ib%2))) & 0xf)
                //         | (((scales_h >> (2*ib)) & 3) << 4);
                let ib_div_2 = ib / 2;
                let ib_mod_2 = ib % 2;

                let ls_low = (block.scales_l[ib_div_2] >> (4 * ib_mod_2)) & 0xF;
                let ls_high = ((block.scales_h >> (2 * ib)) & 0x3) << 4;
                let ls = (ls_low as u16 | ls_high) as i32; // range [0..63]

                // 2b. Compute the scale for this sub-block
                //     In the C code: float dl = d * (ls - 32).
                let dl = d * ((ls - 32) as f32);

                // 2c. Now fill 32 floats of output by reading 16 bytes from qs.
                //     Each byte in qs has two 4-bit indices: low nibble, high nibble.
                //     So we do 16 times:
                //       y[j+0]  = dl * kvalues_iq4nl[ qs[j] & 0xF ];
                //       y[j+16] = dl * kvalues_iq4nl[ qs[j] >> 4 ];
                for j in 0..16 {
                    let byte_val = qs[qs_offset + j];
                    let idx0 = (byte_val & 0xF) as usize; // low nibble
                    let idx1 = (byte_val >> 4) as usize; // high nibble

                    ys[y_offset + j] = dl * KVALUES_IQ4NL[idx0] as f32;
                    ys[y_offset + j + 16] = dl * KVALUES_IQ4NL[idx1] as f32;
                }

                // Advance by 16 bytes in qs, 32 floats in y
                qs_offset += 16;
                y_offset += 32;
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

        for (ys_block, xs_block) in group_for_quantization(xs, ys)? {
            quantize_iq4_nl(
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
