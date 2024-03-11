#![feature(int_roundings)]

use std::path;

use anyhow::Result;
use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, Tensor,
};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use tokenizers::tokenizer::Tokenizer;

use crate::{
    config::Config,
    generate::{generate, GenerateConfig},
    model::{find_multiple, Transformer},
};

mod config;
mod generate;
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
    #[arg(long, default_value = "false")]
    cpu: bool,

    /// The prompt to use
    #[arg(long, default_value = "")]
    prompt: String,

    /// The number of samples to generate.
    #[arg(long, default_value = "5")]
    num_samples: usize,

    /// Max new tokens
    #[arg(long, default_value = "10")]
    max_new_tokens: usize,

    /// The temperature to use.
    #[arg(long, default_value = "0.8")]
    temperature: f64,

    /// The top_k to use.
    #[arg(long)]
    top_k: Option<usize>,

    /// Speculative execution depth
    #[arg(long)]
    speculate_k: Option<usize>,

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
        let api = api.model(args.repo_id.clone());
        let tokenizer_path = api.get("tokenizer.json")?;
        Tokenizer::from_file(tokenizer_path).unwrap()
    };

    // Use bf16 for computation
    let dtype = DType::F32;

    // Use CUDA if available
    let device = if args.cpu {
        Device::Cpu
    } else {
        if cuda_is_available() {
            Device::cuda_if_available(0)?
        } else if metal_is_available() {
            Device::new_metal(0)?
        } else {
            Device::Cpu
        }
    };

    // Varmap and builder, load model from the repo
    let mut varmap = VarMap::new();
    {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model(args.repo_id.clone());
        let model_path = api.get("model.safetensors")?;
        varmap.load(path::Path::new(&model_path))?;
    }
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    // Load the model
    let dim = 2048;
    let intermediate_size = {
        let hidden_dim = dim * 4;
        let n_hidden = (2 * hidden_dim) / (3);
        find_multiple(n_hidden, 256)
    };
    let n_head = 32;
    let n_local_heads = 32;
    let head_dim = dim / n_head;
    let block_size = 1024;
    let mut model = Transformer::load(
        Config {
            vocab_size: tokenizer.get_vocab_size(false),
            dim,
            intermediate_size,
            n_layer: 32,
            n_head,
            n_local_heads,
            head_dim,
            eps: 1e-5,
            block_size,
            max_seq_length: 100,
            rope_base: 10000,
        },
        vb,
    )?;
    println!("finished loading model");

    // Encode the prompt
    let tokens = tokenizer
        .encode("Large language models are ", false)
        .unwrap();
    let tokens = tokens.get_ids().to_vec();
    let prompt = Tensor::new(tokens, &device)?;
    println!("finished encoding prompt");

    // Generate
    for i in 0..args.num_samples {
        let start_gen = std::time::Instant::now();
        let generated_tokens = generate(
            &mut model,
            &prompt,
            GenerateConfig {
                max_new_tokens: args.max_new_tokens,
                temperature: args.temperature,
                top_k: args.top_k,
                speculate_k: args.speculate_k,
                interactive: false,
                block_size,
            },
        )?;
        let dt = start_gen.elapsed().as_secs_f64();
        let tokens_generated = generated_tokens.dims()[0] - prompt.dims()[0];
        let tokens_sec = tokens_generated as f64 / dt;
        println!("Time for inference {i}: {dt} sec total, {tokens_sec} tokens/sec");
    }

    Ok(())
}
