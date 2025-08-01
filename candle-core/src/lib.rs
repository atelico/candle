//! ML framework for Rust
//!
//! ```rust
//! use candle_core::{Tensor, DType, Device};
//! # use candle_core::Error;
//! # fn main() -> Result<(), Error>{
//!
//! let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
//! let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
//! let c = a.matmul(&b)?;
//!
//! # Ok(())}
//! ```
//!
//! ## Features
//!
//! - Simple syntax (looks and feels like PyTorch)
//! - CPU and Cuda backends (and M1 support)
//! - Enable serverless (CPU) small and fast deployments
//! - Model training
//! - Distributed computing (NCCL).
//! - Models out of the box (Llama, Whisper, Falcon, ...)
//!
//! ## FAQ
//!
//! - Why Candle?
//!
//! Candle stems from the need to reduce binary size in order to *enable serverless*
//! possible by making the whole engine smaller than PyTorch very large library volume
//!
//! And simply *removing Python* from production workloads.
//! Python can really add overhead in more complex workflows and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.
//!
//! Rust is cool, and a lot of the HF ecosystem already has Rust crates [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers)
//!
//! ## Other Crates
//!
//! Candle consists of a number of crates. This crate holds core the common data structures but you may wish
//! to look at the docs for the other crates which can be found here:
//!
//! - [candle-core](https://docs.rs/candle-core/). Core Datastructures and DataTypes.
//! - [candle-nn](https://docs.rs/candle-nn/). Building blocks for Neural Nets.
//! - [candle-datasets](https://docs.rs/candle-datasets/). Rust access to commonly used Datasets like MNIST.
//! - [candle-examples](https://docs.rs/candle-examples/). Examples of Candle in Use.
//! - [candle-onnx](https://docs.rs/candle-onnx/). Loading and using ONNX models.
//! - [candle-pyo3](https://docs.rs/candle-pyo3/). Access to Candle from Python.
//! - [candle-transformers](https://docs.rs/candle-transformers/). Candle implemntation of many published transformer models.
//!

#[cfg(feature = "accelerate")]
mod accelerate;
pub mod backend;
pub mod backprop;
pub mod conv;
mod convert;
pub mod cpu;
pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
mod custom_op;
mod device;
pub mod display;
mod dtype;
pub mod dummy_cuda_backend;
pub mod dummy_dtype;
mod dummy_metal_backend;
pub mod error;
mod indexer;
pub mod layout;
#[cfg(feature = "metal")]
pub mod metal_backend;

#[cfg(feature = "mkl")]
mod mkl;
pub mod npy;
pub mod op;
pub mod pickle;
pub mod quantized;
pub mod safetensors;
pub mod scalar;
pub mod shape;
mod sort;
mod storage;
pub mod streaming;
mod strided_index;
mod tensor;
mod tensor_cat;
pub mod test_utils;
pub mod utils;
mod variable;
// regardless of features, we have autoreleasepool,
// dummy implementation does nothing
pub use utils::autoreleasepool;

#[cfg(feature = "cudnn")]
pub use cuda_backend::cudnn;

pub use cpu_backend::{CpuStorage, CpuStorageRef};
#[cfg(feature = "cuda")]
pub use cuda_backend as cuda;
pub use custom_op::{CustomOp1, CustomOp2, CustomOp3, InplaceOp1, InplaceOp2, InplaceOp3, UgIOp1};
pub use device::{Device, DeviceLocation, NdArray};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use error::{Context, Error, Result};
pub use indexer::{IndexOp, TensorIndexer};
pub use layout::Layout;
pub use shape::{Shape, D};
pub use storage::Storage;
pub use streaming::{StreamTensor, StreamingBinOp, StreamingModule};
pub use strided_index::{StridedBlocks, StridedIndex};
use sysinfo::System;
pub use tensor::{from_storage_no_op, Tensor, TensorId};
pub use variable::Var;

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend as cuda;

pub use cuda::{CudaDevice, CudaStorage};

#[cfg(feature = "metal")]
pub use metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(not(feature = "metal"))]
pub use dummy_metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub trait ToUsize2 {
    fn to_usize2(self) -> (usize, usize);
}

impl ToUsize2 for usize {
    fn to_usize2(self) -> (usize, usize) {
        (self, self)
    }
}

impl ToUsize2 for (usize, usize) {
    fn to_usize2(self) -> (usize, usize) {
        self
    }
}

/// Defining a module with forward method using a single argument.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}

impl<M: Module> Module for Option<&M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

/// A single forward method using a single single tensor argument and a flag to
/// separate the training and evaluation behaviors.
pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}

impl<M: Module> ModuleT for M {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        self.forward(xs)
    }
}

/// Amount of available memory in bytes.
pub fn get_memory_allocated(device: &Device) -> Result<usize> {
    match device {
        Device::Cpu => {
            let total_mem = get_total_system_memory(device)?;
            let mut sys = System::new_all();
            sys.refresh_cpu();
            let avail_mem = usize::try_from(sys.available_memory())?;
            Ok(total_mem.saturating_sub(avail_mem) as usize)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            use crate::cuda::cudarc::driver::result;
            use crate::cuda_backend::WrapErr;

            dev.cuda_stream().context().bind_to_thread().w()?;

            let (free, total) = result::mem_get_info().w()?;

            Ok(total - free)
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => {
            crate::bail!("Cannot get memory available for CUDA device")
        }
        #[cfg(feature = "metal")]
        Device::Metal(dev) => {
            let max = dev.recommended_max_working_set_size();
            let alloc = dev.current_allocated_size();
            let avail = max.saturating_sub(alloc);

            #[allow(clippy::cast_possible_truncation)]
            Ok(max.saturating_sub(avail) as usize)
        }
        #[cfg(not(feature = "metal"))]
        Device::Metal(_) => {
            crate::bail!("Cannot get memory available for Metal device")
        }
    }
}

/// Amount of total memory in bytes.
pub fn get_total_system_memory(device: &Device) -> Result<usize> {
    match device {
        Device::Cpu => {
            let mut sys = System::new_all();
            sys.refresh_cpu();
            Ok(usize::try_from(sys.total_memory())?)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            use crate::cuda::cudarc::driver::result;
            use crate::cuda_backend::WrapErr;

            dev.cuda_stream().context().bind_to_thread().w()?;

            let (_free, total) = result::mem_get_info().w()?;

            Ok(total)
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => {
            crate::bail!("Cannot get total memory for CUDA device")
        }
        #[cfg(feature = "metal")]
        #[allow(clippy::cast_possible_truncation)]
        Device::Metal(dev) => {
            const SIZE_IN_MB: usize = 1024 * 1024;

            // Get system RAM in MB
            let system_ram_mb = {
                let mut sys = System::new_all();
                sys.refresh_cpu();
                usize::try_from(sys.total_memory())? / SIZE_IN_MB
            };

            // Check for Metal GPU wired limit
            let metal_cap_mb = std::process::Command::new("sysctl")
                .arg("-n")
                .arg("iogpu.wired_limit_mb")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.trim().parse::<usize>().ok());

            // Apply default cap based on system RAM if not set or 0
            let default_cap = match system_ram_mb {
                x if x <= 36 * 1024 => (system_ram_mb * 2) / 3,
                x if x > 36 * 1024 => (system_ram_mb * 3) / 4,
                x => {
                    return Err(crate::Error::Msg(format!(
                        "Invalid system ram mb value {x}."
                    )))
                }
            };

            let metal_cap_mb = match metal_cap_mb {
                Some(0) => default_cap,
                Some(x) => x,
                None => default_cap,
            };

            let device_max = dev.recommended_max_working_set_size() as usize;
            let metal_cap_bytes = metal_cap_mb * SIZE_IN_MB;

            Ok(device_max.min(metal_cap_bytes))
        }
        #[cfg(not(feature = "metal"))]
        Device::Metal(_) => {
            crate::bail!("Cannot get memory available for Metal device")
        }
    }
}

/// A convenience macro so you can write:
/// ```
/// let result = autorelease_block!({
///     do_heavy_metal_work()?
/// });
/// ```
#[macro_export]
macro_rules! autorelease_block {
    ($body:block) => {{
        let _pool = $crate::utils::autoreleasepool();
        $body
    }};
}

#[macro_export]
macro_rules! autorelease_block_for_device {
    ($device:expr, $body:block) => {{
        let _pool = $crate::utils::autoreleasepool();
        #[cfg(feature = "metal")]
        if let candle_core::Device::Metal(_) = $device {
            // print total memory allocated at time of block
            #[cfg(feature = "metal")]
            use candle_core::get_memory_allocated;
            #[cfg(feature = "metal")]
            use candle_core::Device;
            #[cfg(feature = "metal")]
            println!(
                "Memory allocated before block: {} bytes",
                get_memory_allocated($device).unwrap_or(0)
            );
        }
        $body
    }};
}
