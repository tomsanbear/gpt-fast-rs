use core::panic;
use std::cmp::min;

use crate::model::Transformer;
use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::ops::softmax;
use rand::{distributions::Distribution, SeedableRng};

pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub interactive: bool,
    pub speculate_k: Option<usize>,
    pub block_size: usize,
    pub temperature: f64,
    pub top_k: Option<usize>,
}

pub fn generate(model: &mut Transformer, prompt: &Tensor, cfg: GenerateConfig) -> Result<Tensor> {
    let device = prompt.device();
    let dtype = prompt.dtype();

    let t = prompt.dims()[0];
    let t_new = t + cfg.max_new_tokens;
    let max_seq_len = min(t_new, cfg.block_size);

    let seq = Tensor::zeros(t_new, dtype, device)?;
    let input_pos = Tensor::arange(0, t as u8, device)?;

    let next_token = prefill(model, prompt, input_pos, cfg.temperature, cfg.top_k)?;
    let seq = seq.slice_assign(&[t..t + 1], &seq)?;

    let input_pos = Tensor::new(t as u8, &device)?;

    // TODO: implement speculative execution
    //accept_counts = [0] * (speculate_k + 1)

    // TODO: implement speculative execution
    if cfg.speculate_k.is_some() {
        panic!("speculate_k not implemented");
    };

    let (generated_tokens, _) = decode_n_tokens(
        model,
        &next_token,
        &input_pos,
        cfg.max_new_tokens - 1,
        cfg.temperature,
        cfg.top_k,
    )?;
    let seq = seq.slice_assign(&[t..t + 1], &Tensor::cat(&generated_tokens, 0)?)?;

    Ok(seq)
}

fn decode_one_token(
    model: &mut Transformer,
    x: &Tensor,
    input_pos: &Tensor,
    temperature: f64,
    top_k: Option<usize>,
) -> Result<(Tensor, Tensor)> {
    let logits = model.forward(&x, input_pos)?;
    let (idx_next, probs) = sample(logits, temperature, top_k)?;
    let idx_next = Tensor::new(idx_next, &x.device())?;
    let probs = Tensor::from_vec(probs, x.shape(), x.device())?;
    Ok((idx_next, probs))
}

fn decode_n_tokens(
    model: &mut Transformer,
    cur_token: &Tensor,
    input_pos: &Tensor,
    num_new_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let mut new_tokens = vec![];
    let mut new_probs = vec![];

    let mut cur_token = cur_token.clone();
    let mut input_pos = input_pos.clone();

    for _ in 0..num_new_tokens {
        let one = Tensor::new(1u8, cur_token.device()).unwrap();
        let (next_token, next_prob) =
            decode_one_token(model, &cur_token, &input_pos, temperature, top_k).unwrap();
        cur_token = next_token.clone();
        input_pos = (input_pos + one).unwrap();
        new_tokens.push(next_token);
        new_probs.push(next_prob);
    }

    Ok((new_tokens, new_probs))
}

fn logits_to_probs(logits: Tensor, temperature: f64, top_k: Option<usize>) -> Result<Tensor> {
    let logits = (logits / (temperature.max(1e-5)))?;

    // TODO: implement tensor.topk function
    if top_k.is_some() {
        panic!("top_k not implemented");
    };

    Ok(softmax(&logits, D::Minus1)?)
}

fn sample_multinomial(prs: &Vec<f32>) -> Result<u32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    let distr = rand::distributions::WeightedIndex::new(prs).map_err(candle_core::Error::wrap)?;
    let next_token = distr.sample(&mut rng) as u32;
    Ok(next_token)
}

fn sample(logits: Tensor, temperature: f64, topk: Option<usize>) -> Result<(u32, Vec<f32>)> {
    let probs = logits_to_probs(logits.i((0, ..))?, temperature, topk)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let idx_next = sample_multinomial(&probs)?;
    Ok((idx_next, probs))
}

fn prefill(
    model: &mut Transformer,
    x: &Tensor,
    input_pos: Tensor,
    temperature: f64,
    top_k: Option<usize>,
) -> Result<Tensor> {
    let logits = model.forward(&x, &input_pos)?;
    let (idx_next, _) = sample(logits, temperature, top_k)?;
    let idx_next = Tensor::new(idx_next, &x.device())?;
    Ok(idx_next)
}
