//! Useful functions for checking features.
use std::str::FromStr;

// ---------- iOS / macOS autorelease‑pool glue ----------
#[cfg(feature = "metal")]
use std::ffi::c_void;

#[cfg(feature = "metal")]
use metal::objc::runtime::{objc_autoreleasePoolPop, objc_autoreleasePoolPush};
// -------------------------------------------------------
#[cfg(feature = "metal")]
pub struct AutoreleasePoolGuard {
    pool: *mut c_void,
}

#[cfg(feature = "metal")]
impl Drop for AutoreleasePoolGuard {
    fn drop(&mut self) {
        unsafe {
            objc_autoreleasePoolPop(self.pool);
        }
    }
}

#[cfg(feature = "metal")]
pub fn autoreleasepool() -> AutoreleasePoolGuard {
    unsafe {
        AutoreleasePoolGuard {
            pool: objc_autoreleasePoolPush(),
        }
    }
}

#[cfg(not(feature = "metal"))]
pub struct AutoreleasePoolGuard;

#[cfg(not(feature = "metal"))]
pub fn autoreleasepool() -> AutoreleasePoolGuard {
    AutoreleasePoolGuard
}

pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

pub fn has_accelerate() -> bool {
    cfg!(feature = "accelerate")
}

pub fn has_mkl() -> bool {
    cfg!(feature = "mkl")
}

pub fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}

pub fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

pub fn with_avx() -> bool {
    cfg!(target_feature = "avx")
}

pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

pub fn with_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

pub fn with_f16c() -> bool {
    cfg!(target_feature = "f16c")
}
