#[derive(Clone, Debug)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) dim: usize,
    pub(crate) n_layer: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) n_head: usize,
    pub(crate) n_local_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) eps: f32,
}
