use half::f16;

use crate::Result;

const GROUP_MAX_EPS: f32 = 1e-15;

pub(super) fn nearest_int(v: f32) -> i32 {
    v.round() as i32
}

/// Validates that the input and output are the right size and returns an iterator which maps each
/// input region `xs` to its corresponding output block in `ys`. Each output region is guaranteed
/// to be `T::BLCK_SIZE` long.
pub(super) fn group_for_quantization<'a, 'b, T: super::k_quants::GgmlType>(
    xs: &'b [f32],
    ys: &'a mut [T],
) -> Result<Vec<(&'a mut T, &'b [f32])>> {
    let block_size = T::BLCK_SIZE;
    let dtype = T::DTYPE;

    let expected_blocks = xs.len() / block_size;
    let actual_blocks = ys.len();

    // Validate that the input is the right size
    if expected_blocks != actual_blocks {
        crate::bail!("quantize {dtype:?}: expected {expected_blocks} blocks but only {actual_blocks} were provided!")
    }

    Ok(ys.iter_mut().zip(xs.chunks_exact(block_size)).collect())
}

/// Validates that the input and output are the right size and returns an iterator which maps each
/// input block `xs` to its corresponding output region in `ys`. Each output region is guaranteed
/// to be `T::BLCK_SIZE` long.
pub(super) fn group_for_dequantization<'a, 'b, T: super::k_quants::GgmlType>(
    xs: &'a [T],
    ys: &'b mut [f32],
) -> Result<Vec<(&'a T, &'b mut [f32])>> {
    let block_size = T::BLCK_SIZE;
    let dtype = T::DTYPE;

    let actual_output_len = ys.len();
    let expected_output_len = xs.len() * block_size;
    // Validate that the output is the right size
    if expected_output_len != actual_output_len {
        crate::bail!("dequantize {dtype:?}: ys (len = {actual_output_len}) does not match the expected length of {expected_output_len}!")
    }

    // Zip the blocks and outputs together
    Ok(xs.iter().zip(ys.chunks_exact_mut(block_size)).collect())
}

pub(super) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

pub(super) unsafe fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: *const f32,
    ls: *mut i8,
    rmse_type: i32,
    qw: *const f32,
) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for i in 0..n {
        let x = *x.add(i);
        let ax = x.abs();
        if ax > amax {
            amax = ax;
            max = x;
        }
    }
    if amax == 0. {
        // all zero
        for i in 0..n {
            *ls.add(i) = 0;
        }
        return 0.;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            *ls.add(i) = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let weight_type = rmse_type % 2;
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let x = *x.add(i);
        let l = nearest_int(iscale * x);
        let l = l.clamp(-nmax, nmax - 1);
        *ls.add(i) = (l + nmax) as i8;
        let w = if !qw.is_null() {
            *qw.add(i)
        } else if weight_type == 1 {
            x * x
        } else {
            1.0
        };
        let l = l as f32;
        sumlx += w * x * l;
        suml2 += w * l * l;
    }
    let mut scale = sumlx / suml2;
    let mut best = scale * sumlx;
    for _itry in 0..3 {
        let iscale = 1.0 / scale;
        let mut slx = 0f32;
        let mut sl2 = 0f32;
        let mut changed = false;
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            let l = l.clamp(-nmax, nmax - 1);
            if l + nmax != *ls.add(i) as i32 {
                changed = true;
            }
            let w = if !qw.is_null() {
                *qw.add(i)
            } else if weight_type == 1 {
                x * x
            } else {
                1.0
            };
            let l = l as f32;
            slx += w * x * l;
            sl2 += w * l * l;
        }
        if !changed || sl2 == 0.0 || slx * slx <= best * sl2 {
            break;
        }
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            *ls.add(i) = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        sumlx = slx;
        suml2 = sl2;
        scale = sumlx / suml2;
        best = scale * sumlx;
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let x = *x.add(i);
            let w = if !qw.is_null() {
                *qw.add(i)
            } else if weight_type == 1 {
                x * x
            } else {
                1.0
            };
            let l = *ls.add(i) as i32 - nmax;
            let mut slx = sumlx - w * x * l as f32;
            if slx > 0. {
                let mut sl2 = suml2 - w * l as f32 * l as f32;
                let new_l = nearest_int(x * sl2 / slx);
                let new_l = new_l.clamp(-nmax, nmax - 1);
                if new_l != l {
                    slx += w * x * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if sl2 > 0. && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        *ls.add(i) = (nmax + new_l) as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        scale = sumlx / suml2;
                        best = scale * sumlx;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if rmse_type < 3 {
        return scale;
    }
    for is in -4..4 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1f32 * is as f32) / max;
        let mut sumlx = 0.;
        let mut suml2 = 0.;
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            let l = l.clamp(-nmax, nmax - 1);
            let w = if !qw.is_null() {
                *qw.add(i)
            } else if weight_type == 1 {
                x * x
            } else {
                1.0
            };
            let l = l as f32;
            sumlx += w * x * l;
            suml2 += w * l * l;
        }
        if suml2 > 0. && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let x = *x.add(i);
                let l = nearest_int(iscale * x);
                *ls.add(i) = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L224
/// (scale, min)
pub(super) fn make_qkx1_quants(nmax: i32, ntry: usize, x: &[f32]) -> (f32, f32) {
    let n = x.len();
    let mut l = vec![0; n];
    // Get min/max
    let min = *x
        .iter()
        .take(n)
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&x[0]);
    let max = *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&x[0]);

    // If min == max, all values are the same => nothing to do here
    if max == min {
        return (0.0, 0.0);
    }

    // Ensure min <= 0.0
    let mut min = min.min(0.);

    // Compute scale and inverse scale
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;

    for _ in 0..ntry {
        let mut sumlx = 0.0;
        let mut suml2 = 0;
        let mut did_change = false;

        for (i, value) in x.iter().enumerate().take(n) {
            let li = nearest_int(iscale * (value - min)).clamp(0, nmax);
            let clamped_li = li as u8;
            if clamped_li != l[i] {
                l[i] = clamped_li;
                did_change = true;
            }
            sumlx += (value - min) * li as f32;
            suml2 += li * li;
        }
        scale = sumlx / suml2 as f32;

        let sum: f32 = x
            .iter()
            .take(n)
            .zip(l.iter().take(n))
            .map(|(xi, &li)| xi - scale * li as f32)
            .sum();

        min = sum / n as f32;
        if min > 0.0 {
            min = 0.0;
        }
        iscale = 1.0 / scale;
        if !did_change {
            break;
        }
    }
    (scale, -min)
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L165
pub(super) fn make_q3_quants(x: &[f32], nmax: i32, do_rmse: bool) -> f32 {
    let n = x.len();
    let mut l = vec![0i8; n];

    let mut max = 0.0;
    let mut amax = 0.0;
    for &xi in x.iter().take(n) {
        let ax = xi.abs();
        if ax > amax {
            amax = ax;
            max = xi;
        }
    }

    if amax == 0.0 {
        return 0.0;
    }

    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0.0;
        let mut suml2 = 0.0;
        for i in 0..n {
            let li = (iscale * x[i]).round() as i32;
            let li = li.clamp(-nmax, nmax - 1);
            l[i] = li as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * li as f32;
            suml2 += w * (li * li) as f32;
        }
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let mut slx = sumlx - w * x[i] * l[i] as f32;
                if slx > 0.0 {
                    let mut sl2 = suml2 - w * (l[i] as i32 * l[i] as i32) as f32;
                    let mut new_l = (x[i] * sl2 / slx).round() as i32;
                    new_l = new_l.clamp(-nmax, nmax - 1);
                    if new_l != l[i] as i32 {
                        slx += w * x[i] * new_l as f32;
                        sl2 += w * (new_l * new_l) as f32;
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for li in l.iter_mut() {
            *li += nmax as i8;
        }
        return sumlx / suml2;
    }
    for i in 0..n {
        let li = (iscale * x[i]).round() as i32;
        l[i] = (li.clamp(-nmax, nmax - 1) + nmax) as i8;
    }
    1.0 / iscale
}

// https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/ggml/src/ggml-quants.c#L744
/// (scale, min)
pub(super) fn make_qkx3_quants(
    nmax: i32,
    x: &[f32],
    weights: Option<&[f32]>,
    rmin: f32,
    rdelta: f32,
    nstep: usize,
    use_mad: bool,
) -> (f32, f32) {
    let n = x.len();
    let mut l: [u8; 32] = [0; 32];
    let mut l_aux: [u8; 32] = [0; 32];

    let mut min_val = x[0];
    let mut max_val = x[0];
    let mut sum_w = match weights {
        Some(w) => w[0],
        None => x[0] * x[0],
    };
    let mut sum_x = sum_w * x[0];

    for i in 1..n {
        if x[i] < min_val {
            min_val = x[i];
        }
        if x[i] > max_val {
            max_val = x[i];
        }
        let w = match weights {
            Some(w) => w[i],
            None => x[i] * x[i],
        };
        sum_w += w;
        sum_x += w * x[i];
    }

    if min_val > 0.0 {
        min_val = 0.0;
    }

    if max_val <= min_val {
        return (0.0, -min_val);
    }

    let mut iscale = nmax as f32 / (max_val - min_val);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0;

    for i in 0..n {
        let l_val = nearest_int(iscale * (x[i] - min_val)).clamp(0, nmax) as u8;
        l[i] = l_val;
        let diff = scale * (l_val as f32) + min_val - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        let w = match weights {
            Some(w) => w[i],
            None => x[i] * x[i],
        };
        best_mad += w * diff;
    }

    if nstep < 1 {
        return (scale, -min_val);
    }

    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max_val - min_val);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0, 0.0, 0.0);

        for i in 0..n {
            let l_val = nearest_int(iscale * (x[i] - min_val)).clamp(0, nmax) as u8;
            l_aux[i] = l_val;
            let w = match weights {
                Some(w) => w[i],
                None => x[i] * x[i],
            };
            sum_l += w * l_val as f32;
            sum_l2 += w * (l_val as f32).powi(2);
            sum_xl += w * l_val as f32 * x[i];
        }

        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;

            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }

            let mut mad = 0.0;
            for i in 0..n {
                let diff = this_scale * (l_aux[i] as f32) + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                let w = match weights {
                    Some(w) => w[i],
                    None => x[i] * x[i],
                };
                mad += w * diff;
            }

            if mad < best_mad {
                l.copy_from_slice(&l_aux);
                best_mad = mad;
                scale = this_scale;
                min_val = this_min;
            }
        }
    }

    (scale, -min_val)
}

// https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/ggml/src/ggml-quants.c#L827
pub(super) fn make_qp_quants(
    n: usize,
    nmax: u8,
    x: &[f32],
    l: &mut [u8],
    quant_weights: &[f32],
) -> f32 {
    assert_eq!(x.len(), n);
    assert_eq!(l.len(), n);
    assert_eq!(quant_weights.len(), n);

    let max = x.iter().copied().fold(0.0, f32::max);
    if max == 0.0 {
        l.iter_mut().for_each(|li| *li = 0);
        return 0.0;
    }

    let mut iscale = nmax as f32 / max;
    for (xi, li) in x.iter().zip(l.iter_mut()) {
        *li = nearest_int(iscale * xi) as u8;
    }

    let scale = 1.0 / iscale;
    let mut best_mse = x
        .iter()
        .zip(l.iter())
        .zip(quant_weights.iter())
        .map(|((&xi, &li), &w)| {
            let diff = xi - scale * li as f32;
            w * diff * diff
        })
        .sum::<f32>();

    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;

        let mse = x
            .iter()
            .zip(quant_weights.iter())
            .map(|(&xi, &w)| {
                let mut li = nearest_int(iscale_is * xi) as u8;
                li = li.min(nmax);
                let diff = xi - scale_is * li as f32;
                w * diff * diff
            })
            .sum::<f32>();

        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    let mut sumlx = 0.0;
    let mut suml2 = 0.0;
    for ((xi, li), &w) in x.iter().zip(l.iter_mut()).zip(quant_weights.iter()) {
        let mut li_new = (iscale * xi).round() as u8;
        li_new = li_new.min(nmax);
        *li = li_new;
        sumlx += w * xi * li_new as f32;
        suml2 += w * (li_new as f32).powi(2);
    }

    for _ in 0..5 {
        let mut n_changed = 0;
        for ((xi, li), &w) in x.iter().zip(l.iter_mut()).zip(quant_weights.iter()) {
            let mut slx = sumlx - w * xi * *li as f32;
            let mut sl2 = suml2 - w * (*li as f32).powi(2);
            if slx > 0.0 && sl2 > 0.0 {
                let new_li = (nearest_int(xi * sl2 / slx) as u8).min(nmax);
                if new_li != *li {
                    slx += w * xi * new_li as f32;
                    sl2 += w * (new_li as f32).powi(2);
                    if slx.powi(2) * suml2 > sumlx.powi(2) * sl2 {
                        *li = new_li;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }

    sumlx / suml2
}

fn best_index_int8(n: usize, val: &[i8], x: f32) -> usize {
    if x <= val[0] as f32 {
        return 0;
    }
    if x >= val[n - 1] as f32 {
        return n - 1;
    }
    let mut ml = 0_usize;
    let mut mu = n - 1;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    let dist_low = x - val[mu - 1] as f32;
    let dist_high = val[mu] as f32 - x;
    if dist_low < dist_high {
        mu - 1
    } else {
        mu
    }
}

#[allow(non_snake_case)]
pub(super) fn quantize_row_iq4_nl(
    xs_block: &[f32],
    super_block_size: usize,
    block_size: usize,
    d: &mut f16,
    qs: &mut [u8],
    scales_h: &mut [u16],
    scales_l: &mut [u8],
    values: &[i8],
    quant_weights: Option<&[f32]>,
    ntry: i32,
) {
    let mut sigma2 = 0.0f32;
    for &val in xs_block.iter() {
        sigma2 += val * val;
    }
    sigma2 *= 2.0 / (super_block_size as f32);

    qs.iter_mut().for_each(|x| *x = 0);

    *d = f16::from_f32(0.);

    let mut scales = vec![0.0f32; super_block_size / block_size];
    let mut weight = vec![0.0f32; block_size];
    let mut L = vec![0u8; super_block_size];

    let mut max_scale = 0.0f32;
    let mut amax_scale = 0.0f32;
    let nb = super_block_size / block_size;

    for ib in 0..nb {
        let start = ib * block_size;
        let end = start + block_size;
        let xb = &xs_block[start..end];
        let Lb = &mut L[start..end];

        if let Some(qw) = quant_weights {
            let qw_block = &qw[start..end];
            for j in 0..block_size {
                weight[j] = qw_block[j] * (sigma2 + xb[j] * xb[j]).sqrt();
            }
        } else {
            for j in 0..block_size {
                weight[j] = xb[j] * xb[j];
            }
        }

        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb.iter() {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        let sign = if ntry > 0 { -1.0 } else { 1.0 };
        let mut d = sign * (max / (values[0] as f32));
        let mut id = 1.0 / d;
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;

        for j in 0..block_size {
            let al = id * xb[j];
            let l = best_index_int8(16, &values, al);
            Lb[j] = l as u8;
            let q = values[l] as f32;
            let w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = sumqx / sumq2;
        let mut best = d * sumqx;

        for itry in -ntry..=ntry {
            let itryf = itry as f32;
            id = (itryf + values[0] as f32) / max;
            sumqx = 0.0f32;
            sumq2 = 0.0f32;
            for j in 0..block_size {
                let al = id * xb[j];
                let l = best_index_int8(16, &values, al);
                let q = values[l] as f32;
                let w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if sumq2 > 0. && sumqx * sumqx > best * sumq2 {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }

        scales[ib] = d;
        let abs_d = d.abs();
        if abs_d > amax_scale {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    if nb > 1 {
        let d_f32 = -max_scale / 32.0;
        *d = f16::from_f32(d_f32);

        let id = if d_f32 != 0.0 { 1.0 / d_f32 } else { 0.0 };

        scales_h.iter_mut().for_each(|x| *x = 0);

        for ib in 0..nb {
            let mut l = nearest_int(id * scales[ib]);
            l = l.clamp(-32, 31);

            let dl = d_f32 * (l as f32);
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };

            let start = ib * block_size;
            let end = start + block_size;
            let xb = &xs_block[start..end];
            let Lb = &mut L[start..end];

            for j in 0..block_size {
                let al = idl * xb[j];
                let idx = best_index_int8(16, &values, al);
                Lb[j] = idx as u8;
            }

            l += 32;
            let l_l = (l & 0xF) as u8;
            let l_h = (l >> 4) as u8;

            if ib % 2 == 0 {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            scales_h[ib / 8] |= (l_h as u16) << (2 * (ib as u16 % 8));
        }
    } else {
        *d = f16::from_f32(scales[0]);
        if ntry > 0 {
            let id = if scales[0] != 0.0 {
                1.0 / scales[0]
            } else {
                0.0
            };
            for j in 0..super_block_size {
                let idx = best_index_int8(16, &values, id * xs_block[j]);
                L[j] = idx as u8;
            }
        }
    }

    for i in 0..(super_block_size / 32) {
        for j in 0..16 {
            let l0 = L[32 * i + j];
            let l1 = L[32 * i + 16 + j] << 4;
            qs[16 * i + j] = l0 | l1;
        }
    }
}
