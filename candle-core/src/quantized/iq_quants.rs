use half::f16;

use crate::Result;

use super::{BlockQ8K, GgmlDType, GgmlType, QK_K};

pub const QK4_NL: usize = 32;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
struct BlockIQ4xs {
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
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        todo!()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q8_0
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
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
                *y = f32::round(x * id) as i8
            }
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
