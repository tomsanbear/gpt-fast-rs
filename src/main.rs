#![feature(int_roundings)]

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use tokenizers::tokenizer::Tokenizer;

use crate::{config::Config, model::Transformer};

mod config;
mod model;
mod utils;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The prompt to use
    #[arg(long, default_value = "")]
    prompt: String,

    /// The number of samples to generate.
    #[arg(long, default_value = "5")]
    samples: usize,

    /// Max new tokens
    #[arg(long, default_value = "100")]
    max_new_tokens: usize,

    /// The temperature to use.
    #[arg(long, default_value = "0.8")]
    temperature: f32,

    /// The top_k to use.
    #[arg(long, default_value = "200")]
    top_k: usize,

    /// Speculative execution depth
    #[arg(long, default_value = "5")]
    speculate_k: usize,

    /// Repo id for the model
    #[arg(long, default_value = "TinyLlama/TinyLlama-1.1B-Chat-v1.0")]
    repo_id: String,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // Tokenizer setup
    let tokenizer = {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("hf-internal-testing/llama-tokenizer".to_string());
        let tokenizer_path = api.get("tokenizer.json")?;
        Tokenizer::from_file(tokenizer_path).unwrap()
    };

    // Use bf16 for computation
    let dtype = DType::BF16;

    // Use CUDA if available
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    // Varmap and builder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    // Load the model
    let model = Transformer::load(
        Config {
            vocab_size: tokenizer.get_vocab_size(false),
            dim: 1024,
            n_layer: 24,
            intermediate_size: 4096,
            n_head: 16,
            n_local_heads: 4,
            head_dim: 64,
            eps: 1e-6,
        },
        vb,
    );

    Ok(())
}
