use anyhow::anyhow;
use candle_core::{Tensor, D};
use candle_nn::{ops::softmax, Dropout};

pub struct ScaledDotProductCfg {
    pub is_causal: bool,
    pub need_weights: bool,
    pub average_attn_weights: bool,
    pub force_grouped: bool,
    pub dropout: f32,
}

pub fn scaled_dot_product_gqa(
    query: Tensor,        // (b, n, h, d)
    key: Tensor,          // (b, s, h, d)
    value: Tensor,        // (b, s, h, d)
    mask: Option<Tensor>, // (b, n, s)
    cfg: ScaledDotProductCfg,
) -> Result<(Tensor, Option<Tensor>), anyhow::Error> {
    if query.dims().len() != 4 || key.dims().len() != 4 || value.dims().len() != 4 {
        return Err(anyhow!("Input tensors must have 4 dimensions"));
    };

    // Move sequence length dimension to axis 2, this makes it faster in torch
    // "b n h d -> b h n d"
    let query = query.permute([0, 2, 1, 3])?;

    // "b s h d -> b h s d"
    let key = key.permute([0, 2, 1, 3])?;

    // "b s h d -> b h s d"
    let value = value.permute([0, 2, 1, 3])?;

    // Extract the dimensions
    let (bq, hq, _nq, dq) = query.dims4()?;
    let (bk, hk, nk, dk) = key.dims4()?;
    let (bv, hv, nv, dv) = value.dims4()?;

    // All batch sizes must be equal
    if !(bq == bk && bq == bv) {
        return Err(anyhow!("Batch sizes must be equal"));
    };

    // All dimension sizes must be equal
    if !(dq == dk && dq == dv) {
        return Err(anyhow!("Dimension sizes must be equal"));
    };

    // key and value should have same size in dim 1 and 2
    if nk != nv || hk != hv {
        return Err(anyhow!(
            "Key and value should have same size in dim 1 and 2"
        ));
    };

    // Query heads must be a multiple of kv heads
    if hq % hk != 0 {
        return Err(anyhow!("Query heads must be a multiple of key/value heads"));
    };

    let scale = (*query.dims().last().unwrap() as f64).sqrt();
    let query = (query / scale)?;
    let num_head_groups = hq / hk;

    let similarity = match num_head_groups > 1 || cfg.force_grouped {
        true => {
            // query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
            // similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
            let (batch_size, heads, seq_len, depth) = query.dims4()?;
            let heads = heads / num_head_groups; // Calculate the number of heads per group.

            // Reshape query to [batch, num_head_groups, heads, seq_len, depth]
            let query_reshaped =
                query.reshape((batch_size, num_head_groups, heads, seq_len, depth))?;

            let query_for_matmul = query_reshaped.sum(1)?;

            // Transpose the last two dimensions of key to align them for matmul.
            let key_transposed = key.transpose(D::Minus2, D::Minus1)?; // [batch, heads, depth, seq_len]

            // Perform batched matrix multiplication.
            query_for_matmul.matmul(&key_transposed.contiguous()?)
        }
        false => {
            // If the number of query/key heads is equal, we can skip grouping the queries,
            // and just use the standard sdot product attention.
            // einsum(query, key, "b h n d, b h s d -> b h n s")
            let query = query.unsqueeze(3)?;
            let key_t = key.transpose(D::Minus2, D::Minus1)?;
            query.matmul(&key_t)
        }
    }?;

    // Expand mask to match the shape of the attn matrix
    let mask = match mask {
        Some(mask) => Some({
            if mask.shape().dims().len() == 2 {
                mask.unsqueeze(1)?.unsqueeze(2)?
            } else if mask.dims().len() == 3 {
                mask.unsqueeze(1)?
            } else {
                mask
            }
        }),
        None => None,
    };

    let similarity = match mask {
        Some(mask) => {
            let on_true = Tensor::zeros_like(&similarity)?;
            mask.where_cond(&on_true, &similarity)?
        }
        None => similarity,
    };

    // attention = F.softmax(similarity / scale, dim=-1)
    let attention = softmax(&(similarity / scale)?, D::Minus1)?;

    // apply dropout
    let attention = match cfg.dropout > 0.0 {
        true => {
            // Original python code:
            // attention = F.dropout(attention, p=dropout, training=self.training)
            Dropout::new(cfg.dropout).forward(&attention, false)?
        }
        false => attention,
    };

    // Apply attention matrix to the value Tensor.
    // out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    let out = attention.matmul(&value.contiguous()?)?;

    // Move head dimension back to axis 2
    // out = rearrange(out, "b h n d -> b n h d")
    let out = out.permute([0, 2, 1, 3])?;

    let attn_weights = match cfg.need_weights {
        false => None,
        true => {
            // Move the sequence dimensions back to positions 1, 2.  Move the head dimension
            // to position 3.  This more closely matches the return shape of the attention
            // output: (b, n, h, d).
            // python code:
            // attn_weights = rearrange(attention, "b h n s -> b n s h")
            let attn_weights = attention.permute([0, 2, 3, 1])?;
            // if average_attn_weights:
            //   attn_weights = attn_weights.mean(dim=1)
            if cfg.average_attn_weights {
                let attn_weights = attn_weights.mean_keepdim(1)?;
                Some(attn_weights)
            } else {
                Some(attn_weights)
            }
        }
    };

    Ok((out, attn_weights))
}
