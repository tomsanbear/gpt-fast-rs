use crate::config::Config;
use crate::utils::{scaled_dot_product_gqa, ScaledDotProductCfg};
use anyhow::{Ok, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, Embedding, Init, Linear, Module, VarBuilder};

struct TransformerBlock {
    attention: Attention,
    ffn: FeedForward,
    ffn_norm: RMSNorm,
    attn_norm: RMSNorm,
}

impl TransformerBlock {
    pub fn load(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let attention = Attention::load(cfg.clone(), vb.pp("attn"))?;
        let ffn = FeedForward::load(cfg.clone(), vb.pp("ffn"))?;
        let ffn_norm = RMSNorm::load(cfg.dim, cfg.eps, vb.pp("ffn_norm"))?;
        let attn_norm = RMSNorm::load(cfg.dim, cfg.eps, vb.pp("attn_norm"))?;
        Ok(Self {
            attention,
            ffn,
            ffn_norm,
            attn_norm,
        })
    }
}

pub struct Transformer {
    tok_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
}

impl Transformer {
    pub fn load(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let tok_embeddings = embedding(cfg.vocab_size, cfg.dim, vb.pp("tok_embeddings"))?;
        let layers: Vec<_> = (0..cfg.n_layer)
            .map(move |i| TransformerBlock::load(cfg.clone(), vb.pp(format!("layer.{i}"))).unwrap())
            .collect();

        Ok(Self {
            tok_embeddings,
            layers,
        })
    }
}

struct KVCache {
    k_cache: Tensor,
    v_cache: Tensor,
}

impl KVCache {
    pub fn load(
        max_batch_size: usize,
        max_seq_length: usize,
        n_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim);
        let k_cache = Tensor::zeros(cache_shape, dtype, device)?;
        let v_cache = Tensor::zeros(cache_shape, dtype, device)?;
        Ok(Self { k_cache, v_cache })
    }

    pub fn update(
        &mut self,
        input_pos: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let k_shape = k.dims4()?;
        let k_out = self.k_cache.slice_assign(
            &[
                0..k_shape.0,
                0..k_shape.1,
                0..k_shape.2,
                0..k_shape.3,
                input_pos,
            ],
            k,
        )?;
        let v_shape = v.dims4()?;
        let v_out = self.v_cache.slice_assign(
            &[
                0..v_shape.0,
                0..v_shape.1,
                0..v_shape.2,
                0..v_shape.3,
                input_pos,
            ],
            v,
        )?;
        Ok((k_out, v_out))
    }
}

struct RMSNorm {
    eps: f32,
    weight: Tensor,
}

impl RMSNorm {
    pub fn load(dim: usize, eps: f32, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", Init::Const(1.0))?;
        Ok(Self { eps, weight })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x
            .div(
                &x.mul(&x)?
                    .mean_keepdim(D::Minus1)?
                    .broadcast_add(&Tensor::new(self.eps, &x.device())?)?
                    .sqrt()?,
            )?
            .mul(&self.weight)?;
        Ok(x)
    }
}

struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    pub fn load(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let w1 = linear(cfg.dim, cfg.intermediate_size, vb.pp("w1"))?;
        let w2 = linear(cfg.intermediate_size, cfg.dim, vb.pp("w2"))?;
        let w3 = linear(cfg.dim, cfg.intermediate_size, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x.apply(&self.w1)?.silu()? * x.apply(&self.w3)?)?.apply(&self.w2)?;
        Ok(x)
    }
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    n_head: usize,
    head_dim: usize,
    n_local_heads: usize,
    dim: usize,
    kv_cache: Option<KVCache>,
}

impl Attention {
    pub fn load(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let total_head_dim = (cfg.n_head + 2 * cfg.n_local_heads) * cfg.head_dim;
        let wqkv = linear_no_bias(cfg.dim, total_head_dim, vb.pp("wqkv"))?;
        let wo = linear_no_bias(cfg.dim, cfg.dim, vb.pp("wo"))?;
        Ok(Self {
            wqkv,
            wo,
            n_head: cfg.n_head,
            head_dim: cfg.head_dim,
            n_local_heads: cfg.n_local_heads,
            dim: cfg.dim,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        freqs_cis: &Tensor,
        mask: Tensor,
        input_pos: Option<Tensor>,
    ) -> Result<Tensor> {
        let (bsz, seqlen, _) = x.dims3()?;

        let kv_size = self.n_local_heads * self.head_dim;
        let q = x.i((.., .., 0..self.dim))?;
        let k = x.i((.., .., self.dim..(self.dim + kv_size)))?;
        let v = x.i((.., .., (self.dim + kv_size)..))?;

        let q = q.reshape((bsz, seqlen, self.n_head, self.head_dim))?;
        let k = k.reshape((bsz, seqlen, self.n_local_heads, self.head_dim))?;
        let v = v.reshape((bsz, seqlen, self.n_local_heads, self.head_dim))?;

        let q = apply_rotary_emb(&q, &freqs_cis)?;
        let k = apply_rotary_emb(&k, &freqs_cis)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let (k, v) = if let Some(kv_cache) = &mut self.kv_cache {
            let (k, v) = kv_cache.update(&input_pos.unwrap(), &k, &v)?;
            (k, v)
        } else {
            (k, v)
        };

        let k = repeat_interleave(&k, self.n_head.div_floor(self.n_local_heads), 1)?;
        let v = repeat_interleave(&v, self.n_head.div_floor(self.n_local_heads), 1)?;
        let (y, _) = scaled_dot_product_gqa(
            q,
            k,
            v,
            Some(mask),
            ScaledDotProductCfg {
                is_causal: false,
                need_weights: false,
                average_attn_weights: false,
                force_grouped: false,
                dropout: 0.0,
            },
        )?;

        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bsz, seqlen, self.dim))?;
        let y = self.wo.forward(&y)?;
        Ok(y)
    }
}

fn outer(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let x = x.unsqueeze(D::Minus1)?;
    let y = y.unsqueeze(0)?;
    Ok((x * y)?)
}

fn polar(abs: &Tensor, angle: &Tensor) -> Result<(Tensor, Tensor)> {
    let real = (abs * angle.cos()?)?;
    let imag = (abs * angle.sin()?)?;
    Ok((real, imag))
}

fn precompute_freqs_cis(seq_len: u8, n_elem: u8, base: u8, device: &Device) -> Result<Tensor> {
    let freqs = (1.0
        / Tensor::new(base, &device)?.broadcast_pow(
            &Tensor::arange_step(0, n_elem, 2, device)?.i(0..(n_elem.div_floor(2).into()))?,
        )?)?;
    let t = Tensor::arange(0, seq_len, &device)?;
    let freqs = outer(&t, &freqs)?;
    let freqs_cis = polar(&freqs.ones_like()?, &freqs)?;
    let cache = Tensor::stack(&[freqs_cis.0, freqs_cis.1], D::Minus1)?;
    Ok(cache)
}

fn apply_rotary_emb(x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
    let x_shaped = x.reshape((x.dims4()?.0, x.dims4()?.1, x.dims4()?.2 - 1, 2))?;
    let freqs_cis = freqs_cis.reshape((1, x_shaped.dims4()?.1, 1, x_shaped.dims4()?.3, 2))?;
    let x_out_2 = Tensor::stack(
        &[
            ((x_shaped.i((.., .., .., 0))? * freqs_cis.i((.., .., .., 0)))?
                - (x_shaped.i((.., .., .., 1))? * freqs_cis.i((.., .., .., 1)))?)?,
            ((x_shaped.i((.., .., .., 1))? * freqs_cis.i((.., .., .., 0)))?
                - (x_shaped.i((.., .., .., 0))? * freqs_cis.i((.., .., .., 1)))?)?,
        ],
        D::Minus1,
    )?;
    let x_out_2 = x_out_2.flatten(3, D::Minus1)?;
    Ok(x_out_2)
}

fn repeat_interleave(x: &Tensor, n: usize, dim: usize) -> Result<Tensor> {
    let x = x.unsqueeze(1)?;
    let x = x.repeat(&[1, n, 1])?;
    Ok(x.flatten(1, D::Minus1)?)
}