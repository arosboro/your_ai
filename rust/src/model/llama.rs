use mlx_macros::ModuleParameters as DeriveModuleParameters;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nested::NestedHashMap;
use mlx_rs::nn::{Embedding, Linear, QuantizedLinear, RmsNorm};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Wrapper for Linear layer that can be either F16 (standard) or Quantized (4-bit)
#[derive(Debug, Clone)]
pub enum LinearLayer {
    F16(Linear),
    Quantized(QuantizedLinear),
    Skeleton(Linear), // Placeholder for 1x1 init
}

impl LinearLayer {
    pub fn new_skeleton() -> Result<Self, Exception> {
        // Create 1x1 linear layer
        let mut l = Linear::new(1, 1)?;
        // Reset weight to 1x1 to be sure
        let w = Array::from_slice(&[0.0f32], &[1, 1]);
        let b = Array::from_slice(&[0.0f32], &[1]);
        *l.weight = w;
        *l.bias = Some(b);
        Ok(LinearLayer::Skeleton(l))
    }

}

impl Module<Array> for LinearLayer {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: Array) -> Result<Self::Output, Self::Error> {
        match self {
            LinearLayer::F16(l) => l.forward(&x),
            LinearLayer::Quantized(l) => l.forward(&x),
            LinearLayer::Skeleton(l) => l.forward(&x), // Should expect shape mismatch if used before loading

        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            LinearLayer::F16(l) => l.training_mode(mode),
            LinearLayer::Quantized(l) => l.training_mode(mode),
            LinearLayer::Skeleton(l) => l.training_mode(mode),

        }
    }
}

impl ModuleParameters for LinearLayer {
    fn parameters(&self) -> NestedHashMap<std::rc::Rc<str>, &Array> {
        match self {
            LinearLayer::F16(l) => l.parameters(),
            LinearLayer::Quantized(l) => l.parameters(),
            LinearLayer::Skeleton(l) => l.parameters(),

        }
    }

    fn parameters_mut(&mut self) -> NestedHashMap<std::rc::Rc<str>, &mut Array> {
        match self {
            LinearLayer::F16(l) => l.parameters_mut(),
            LinearLayer::Quantized(l) => l.parameters_mut(),
            LinearLayer::Skeleton(l) => l.parameters_mut(),

        }
    }

    fn trainable_parameters(&self) -> NestedHashMap<std::rc::Rc<str>, &Array> {
        match self {
            LinearLayer::F16(l) => l.trainable_parameters(),
            LinearLayer::Quantized(l) => l.trainable_parameters(),
            LinearLayer::Skeleton(l) => l.trainable_parameters(),

        }
    }

    fn num_parameters(&self) -> usize {
        match self {
            LinearLayer::F16(l) => l.num_parameters(),
            LinearLayer::Quantized(l) => l.num_parameters(),
            LinearLayer::Skeleton(l) => l.num_parameters(),

        }
    }

    fn freeze_parameters(&mut self, freeze: bool) {
        match self {
            LinearLayer::F16(l) => l.freeze_parameters(freeze),
            LinearLayer::Quantized(l) => l.freeze_parameters(freeze),
            LinearLayer::Skeleton(l) => l.freeze_parameters(freeze),

        }
    }

    fn unfreeze_parameters(&mut self, freeze: bool) {
        match self {
            LinearLayer::F16(l) => l.unfreeze_parameters(freeze),
            LinearLayer::Quantized(l) => l.unfreeze_parameters(freeze),
            LinearLayer::Skeleton(l) => l.unfreeze_parameters(freeze),

        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            LinearLayer::F16(l) => l.all_frozen(),
            LinearLayer::Quantized(l) => l.all_frozen(),
            LinearLayer::Skeleton(l) => l.all_frozen(),

        }
    }

    fn any_frozen(&self) -> Option<bool> {
         match self {
            LinearLayer::F16(l) => l.any_frozen(),
            LinearLayer::Quantized(l) => l.any_frozen(),
            LinearLayer::Skeleton(l) => l.any_frozen(),

        }
    }
}


/// Llama model configuration parsed from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_hidden_layers: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<EosToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EosToken {
    Single(i32),
    Multiple(Vec<i32>),
}

impl LlamaConfig {
    pub fn from_json(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Estimate total model parameters
    pub fn estimate_num_parameters(&self) -> u64 {
        // Embedding layer
        let embedding_params = (self.vocab_size * self.hidden_size) as u64;

        // Each transformer layer has:
        // - Attention: 4 projections (q, k, v, o)
        // - MLP: gate_proj + up_proj + down_proj
        // - Layer norms
        let attention_params_per_layer = (
            // q_proj
            (self.hidden_size * self.num_attention_heads * (self.hidden_size / self.num_attention_heads)) +
            // k_proj and v_proj
            2 * (self.hidden_size * self.num_key_value_heads * (self.hidden_size / self.num_attention_heads)) +
            // o_proj
            (self.num_attention_heads * (self.hidden_size / self.num_attention_heads) * self.hidden_size)
        ) as u64;

        let mlp_params_per_layer = (
            // gate_proj + up_proj (both go to intermediate_size)
            2 * (self.hidden_size * self.intermediate_size) +
            // down_proj
            (self.intermediate_size * self.hidden_size)
        ) as u64;

        // RMS norms (2 per layer: pre-attention and pre-mlp)
        let norm_params_per_layer = (2 * self.hidden_size) as u64;

        let params_per_layer =
            attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer;
        let total_layer_params = params_per_layer * self.num_hidden_layers as u64;

        // Final layer norm + output projection
        let output_params = (self.hidden_size + self.vocab_size * self.hidden_size) as u64;

        embedding_params + total_layer_params + output_params
    }

    /// Estimate memory requirements in bytes (FP16)
    pub fn estimate_memory_bytes(&self) -> u64 {
        let num_params = self.estimate_num_parameters();
        // FP16: 2 bytes per parameter
        // Add 50% overhead for activations, gradients (for LoRA), optimizer states
        let base_memory = num_params * 2;
        (base_memory as f64 * 1.5) as u64
    }

    /// Estimate memory requirements in GB
    pub fn estimate_memory_gb(&self) -> f64 {
        self.estimate_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Check if model is safe to load given available memory
    pub fn check_memory_safety(
        &self,
        available_gb: f64,
        safety_margin_gb: f64,
    ) -> Result<(), String> {
        let required_gb = self.estimate_memory_gb();
        let safe_limit = available_gb - safety_margin_gb;

        if required_gb > safe_limit {
            Err(format!(
                "Model requires ~{:.1} GB but only {:.1} GB available (with {:.1} GB safety margin). \
                Model is too large for this system.",
                required_gb, safe_limit, safety_margin_gb
            ))
        } else {
            Ok(())
        }
    }

    /// Print memory estimation report
    pub fn print_memory_estimate(&self, system_memory_gb: f64) {
        let num_params = self.estimate_num_parameters();
        let required_gb = self.estimate_memory_gb();
        let percentage = (required_gb / system_memory_gb) * 100.0;

        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Model Memory Estimation");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  Parameters:        {:.2}B ({} total)",
            num_params as f64 / 1_000_000_000.0,
            num_params
        );
        println!("  Estimated memory:  {:.1} GB", required_gb);
        println!("  System memory:     {:.1} GB", system_memory_gb);
        println!("  Usage:             {:.1}%", percentage);

        if percentage > 80.0 {
            println!("  Status:            ⚠️  UNSAFE - Model too large!");
            println!("\n  Recommendation: Use a smaller model (8B-13B recommended)");
        } else if percentage > 60.0 {
            println!("  Status:            ⚠️  CAUTION - High memory usage");
            println!("\n  Recommendation: Monitor memory closely during training");
        } else {
            println!("  Status:            ✓ SAFE");
        }
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    }
}

/// Grouped Query Attention for Llama
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LlamaAttention {
    pub config: LlamaConfig,
    #[param]
    pub q_proj: LinearLayer,
    #[param]
    pub k_proj: LinearLayer,
    #[param]
    pub v_proj: LinearLayer,
    #[param]
    pub o_proj: LinearLayer,
    #[param]
    pub rope: RotaryEmbedding,
    pub head_dim: i32,
    pub num_kv_groups: i32,
    #[param]
    pub q_proj_lora: Option<LoraAdapter>,
    #[param]
    pub k_proj_lora: Option<LoraAdapter>,
    #[param]
    pub v_proj_lora: Option<LoraAdapter>,
    #[param]
    pub o_proj_lora: Option<LoraAdapter>,
    // Removed #[param(skip)] as it likely causes derive issues or KVCache not implementing traits
    pub kv_cache: Option<KVCache>,
}

#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LoraAdapter {
    #[param]
    pub lora_a: Linear,
    #[param]
    pub lora_b: Linear,
    pub scale: f32,
    pub rank: usize,
    pub dropout: f32,
}

/// Custom Rotary Embedding with Offset Support
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    _dim: i32,
    base: f32,
    // We don't store precomputed cache here to keep it simple with MLX graphs
    // But we could optimize later.
}

impl RotaryEmbedding {
    pub fn new(dim: i32, base: f32) -> Self {
        Self { _dim: dim, base }
    }

    pub fn forward(&self, x: &Array, offset: usize) -> Result<Array, Exception> {
        // x: [B, H, L, D] or [B, L, H, D] - check usage
        // usage in LlamaAttention:
        // q: [B, L, num_heads, head_dim] -> transpose -> [B, num_heads, L, head_dim]
        // rope called on [B, num_heads, L, head_dim] (based on transpose axes 0, 2, 1, 3?? No wait)

        // Let's re-read LlamaAttention::forward carefully:
        // q = q.reshape(&[B, L, n_h, h_d])
        // k = k.reshape(&[B, L, n_kv, h_d])
        // rope.forward(&q)

        // This means rope receives [B, L, H, D].
        // seq_dim is 1.

        let seq_len = x.dim(1);
        let head_dim = x.dim(3);

        // Generate freqs
        // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        let half_dim = head_dim / 2;
        // Fix: Provide 2 generic args <f32, f32>
        let start = Array::arange::<f32, f32>(0.0, half_dim as f32, 1.0)?; // [0, 1, ..., half-1]
        let div_term = start.multiply(Array::from_f32(2.0))?.divide(Array::from_f32(head_dim as f32))?; // (2i/dim)
        // base^(-2i/dim) = exp(-2i/dim * ln(base))
        // or just pow
        // Let's use exp approach: exp(-2i/d * ln(base))
        let ln_base = self.base.ln();
        let exponent = div_term.multiply(Array::from_f32(-ln_base))?;
        let inv_freq = mlx_rs::ops::exp(&exponent)?; // [half_dim]

        // positions: arange(offset, offset+seq_len)
        // Fix: Provide 2 generic args <f32, f32>
        let positions = Array::arange::<f32, f32>(
            offset as f32,
            (offset + seq_len as usize) as f32,
            1.0
        )?; // [seq_len]

        // freqs = outer(positions, inv_freq) -> [seq_len, half_dim]
        // MLX doesn't have outer, use broadcast multiply
        // positions: [seq_len, 1], inv_freq: [1, half_dim]
        let positions_col = positions.reshape(&[seq_len, 1])?;
        let inv_freq_row = inv_freq.reshape(&[1, half_dim])?;
        let freqs = positions_col.matmul(&inv_freq_row)?; // [seq_len, half_dim]
        let emb = {
             // Transpose to use axis 0 for concatenation
             // [L, D/2] -> [D/2, L]
             let t1 = freqs.transpose_axes(&[1, 0])?;
             let t2 = freqs.transpose_axes(&[1, 0])?;
             let c = mlx_rs::ops::concatenate(&[&t1, &t2])?;

             // [D, L] -> [L, D]
             // Fix: Explicitly reshape to [head_dim, seq_len] because mlx_rs::concatenate may flatten if axis is default
             let c = c.reshape(&[head_dim, seq_len])?;
             c.transpose_axes(&[1, 0])?
        };

        // Expand for broadcasting against x: [B, L, H, D]
        // emb is [L, D]. We need [1, L, 1, D]
        let emb_broadcast = emb.reshape(&[1, seq_len, 1, head_dim])?;

        let cos = mlx_rs::ops::cos(&emb_broadcast)?;
        let sin = mlx_rs::ops::sin(&emb_broadcast)?;

        // Apply rotation
        // x_rot = [-x[..., half:], x[..., :half]]
        // Use split instead of slice as slice API is unsure
        // Split x into 2 parts along last axis (-1 equivalent to 3 for [B, L, H, D])
        // Assuming split takes simple integer for equal splits
        let parts = mlx_rs::ops::split(x, 2, -1)?;
        let x_first = &parts[0]; // x[..., :half]
        let x_second = &parts[1]; // x[..., half:]

        let neg_x_second = x_second.multiply(Array::from_f32(-1.0))?;

        // rotate_x = cat(-x2, x1, -1)
        let rotate_x = {
             // Transpose last axis to 0 for concat (workaround for axis support)
             // [B, L, H, D/2] -> [D/2, B, L, H]
             // Note: using explicit axis indices for transpose, check rank
             let t1 = neg_x_second.transpose_axes(&[3, 0, 1, 2])?;
             let t2 = x_first.transpose_axes(&[3, 0, 1, 2])?;
             let c = mlx_rs::ops::concatenate(&[&t1, &t2])?;

             // Fix: Explicitly reshape because mlx_rs::concatenate returns flattened array
             let b = x.dim(0);
             let l = x.dim(1);
             let h = x.dim(2);
             let c = c.reshape(&[head_dim, b, l, h])?;

             // [D, B, L, H] -> [B, L, H, D]
             c.transpose_axes(&[1, 2, 3, 0])?
        };

        // output = (x * cos) + (rotate_x * sin)
        let term1 = x.multiply(&cos)?;
        let term2 = rotate_x.multiply(&sin)?;

        term1.add(&term2)
    }
}

// Module Impl for RotaryEmbedding (empty params)
impl Module<Array> for RotaryEmbedding {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: Array) -> Result<Self::Output, Self::Error> {
        RotaryEmbedding::forward(self, &x, 0)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

impl ModuleParameters for RotaryEmbedding {
    fn parameters(&self) -> NestedHashMap<std::rc::Rc<str>, &Array> {
        NestedHashMap::new()
    }
    fn parameters_mut(&mut self) -> NestedHashMap<std::rc::Rc<str>, &mut Array> {
        NestedHashMap::new()
    }
    fn trainable_parameters(&self) -> NestedHashMap<std::rc::Rc<str>, &Array> {
        NestedHashMap::new()
    }
    fn num_parameters(&self) -> usize { 0 }
    fn freeze_parameters(&mut self, _freeze: bool) {}
    fn unfreeze_parameters(&mut self, _freeze: bool) {}
    fn all_frozen(&self) -> Option<bool> { Some(true) }
    fn any_frozen(&self) -> Option<bool> { Some(true) }
}

/// Key-Value Cache for Autoregressive Generation
#[derive(Debug, Clone)]
pub struct KVCache {
    pub key_cache: Array,
    pub value_cache: Array,
    pub offset: usize,
}

impl KVCache {
    /// Create a new cache for a specific layer
    pub fn new() -> Self {
        Self {
            key_cache: Array::from_slice::<f32>(&[], &[0]),
            value_cache: Array::from_slice::<f32>(&[], &[0]),
            offset: 0,
        }
    }
}

impl LoraAdapter {
    pub fn new(in_features: i32, out_features: i32, rank: usize, alpha: f32, dropout: f32) -> Result<Self, Exception> {
        // Initialize A with uniform distribution
        let k = 1.0 / (rank as f32).sqrt();
        let mut lora_a = Linear::new(in_features, rank as i32)?;
        // lora_a needs specific init
        let weight_a = mlx_rs::random::uniform::<_, f32>(-k, k, &[rank as i32, in_features], None)?;
        *lora_a.weight = weight_a;
        *lora_a.bias = None;

        // Initialize B with zeros
        let mut lora_b = Linear::new(rank as i32, out_features)?;
        let weight_b = mlx_rs::ops::zeros::<f32>(&[out_features, rank as i32])?;
        *lora_b.weight = weight_b;
        *lora_b.bias = None;

        Ok(Self {
            lora_a,
            lora_b,
            scale: alpha / (rank as f32),
            rank,
            dropout,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let a = self.lora_a.forward(x)?;
        let b = self.lora_b.forward(&a)?;
        b.multiply(Array::from_f32(self.scale))
    }

    pub fn new_skeleton(rank: usize, alpha: f32, dropout: f32) -> Result<Self, Exception> {
        // 1x1 skeleton
        let mut lora_a = Linear::new(1, 1)?;
        *lora_a.weight = Array::from_slice(&[0.0f32], &[1, 1]);
        *lora_a.bias = None;

        let mut lora_b = Linear::new(1, 1)?;
        *lora_b.weight = Array::from_slice(&[0.0f32], &[1, 1]);
        *lora_b.bias = None;

        Ok(Self {
            lora_a,
            lora_b,
            scale: alpha / (rank as f32),
            rank,
            dropout,
        })
    }
}

impl LlamaAttention {
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let num_kv_groups = config.num_attention_heads / config.num_key_value_heads;

        let mut q_proj = LinearLayer::F16(Linear::new(config.hidden_size, config.num_attention_heads * head_dim)?);
        let mut k_proj = LinearLayer::F16(Linear::new(config.hidden_size, config.num_key_value_heads * head_dim)?);
        let mut v_proj = LinearLayer::F16(Linear::new(config.hidden_size, config.num_key_value_heads * head_dim)?);
        let mut o_proj = LinearLayer::F16(Linear::new(config.num_attention_heads * head_dim, config.hidden_size)?);

        // Disable biases if configured (default for Llama 3)
        if !config.attention_bias {
            if let LinearLayer::F16(l) = &mut q_proj { *l.bias = None; }
            if let LinearLayer::F16(l) = &mut k_proj { *l.bias = None; }
            if let LinearLayer::F16(l) = &mut v_proj { *l.bias = None; }
            if let LinearLayer::F16(l) = &mut o_proj { *l.bias = None; }
        }

        let rope = RotaryEmbedding::new(head_dim, config.rope_theta); // Custom RoPE

        Ok(Self {
            config: config.clone(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            head_dim,
            num_kv_groups,
            q_proj_lora: None,
            k_proj_lora: None, // Fix: Added missing field
            v_proj_lora: None,
            o_proj_lora: None,
            // Non-parametric state
            kv_cache: None,
        })
    }

    pub fn new_skeleton(config: &LlamaConfig) -> Result<Self, Exception> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let num_kv_groups = config.num_attention_heads / config.num_key_value_heads;

        // Skeleton projections
        let q_proj = LinearLayer::new_skeleton()?;
        let k_proj = LinearLayer::new_skeleton()?;
        let v_proj = LinearLayer::new_skeleton()?;
        let o_proj = LinearLayer::new_skeleton()?;

        // Rope needs actual head_dim to work logically if called, but uses no memory
        let rope = RotaryEmbedding::new(head_dim, config.rope_theta);

        Ok(Self {
            config: config.clone(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            head_dim,
            num_kv_groups,
            q_proj_lora: None,
            k_proj_lora: None, // Fix: Added missing field
            v_proj_lora: None,
            o_proj_lora: None,
            kv_cache: None,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let (batch_size, seq_len, _) = (x.dim(0), x.dim(1), x.dim(2));

        // Project to Q, K, V
        let mut q = self.q_proj.forward(x.clone())?;
        if let Some(lora) = &mut self.q_proj_lora {
            let lora_out = lora.forward(x)?;
            q = q.add(&lora_out)?;
        }

        let mut k = self.k_proj.forward(x.clone())?;
        if let Some(lora) = &mut self.k_proj_lora {
            let lora_out = lora.forward(x)?;
            k = k.add(&lora_out)?;
        }

        let mut v = self.v_proj.forward(x.clone())?;
        if let Some(lora) = &mut self.v_proj_lora {
            let lora_out = lora.forward(x)?;
            v = v.add(&lora_out)?;
        }

        // Reshape for multi-head attention
        // Q: [B, L, num_heads * head_dim] -> [B, L, num_heads, head_dim]
        q = q.reshape(&[
            batch_size,
            seq_len,
            self.config.num_attention_heads,
            self.head_dim,
        ])?;
        k = k.reshape(&[
            batch_size,
            seq_len,
            self.config.num_key_value_heads,
            self.head_dim,
        ])?;
        v = v.reshape(&[
            batch_size,
            seq_len,
            self.config.num_key_value_heads,
            self.head_dim,
        ])?;

        // Apply RoPE to Q and K
        // Need offset for RoPE if caching
        let offset = if let Some(cache) = &self.kv_cache {
            cache.offset
        } else {
            0
        };

        q = self.rope.forward(&q, offset)?;
        k = self.rope.forward(&k, offset)?;

        // Update KV Cache
        if let Some(cache) = &mut self.kv_cache {
            // Concatenate with existing cache if not empty
            if cache.offset > 0 {
                // K: [B, L_new, n_kv, d] -> [B, L_total, n_kv, d]
                // Transpose axis 1 (seq_len) to 0 for concatenation
                let k_new = {
                    let t_cache = cache.key_cache.transpose_axes(&[1, 0, 2, 3])?;
                    let t_curr = k.transpose_axes(&[1, 0, 2, 3])?;
                    let c = mlx_rs::ops::concatenate(&[&t_cache, &t_curr])?;

                    // Fix: Reshape flattened result
                    let total_len = t_cache.dim(0) + t_curr.dim(0);
                    let b = t_cache.dim(1); // num_kv_heads or batch? check transpose. [1,0,2,3] -> [L, B, H, D]
                    let h = t_cache.dim(2);
                    let d = t_cache.dim(3);
                    let c = c.reshape(&[total_len, b, h, d])?;

                    c.transpose_axes(&[1, 0, 2, 3])?
                };

                let v_new = {
                    let t_cache = cache.value_cache.transpose_axes(&[1, 0, 2, 3])?;
                    let t_curr = v.transpose_axes(&[1, 0, 2, 3])?;
                    let c = mlx_rs::ops::concatenate(&[&t_cache, &t_curr])?;

                    // Fix: Reshape flattened result
                    let total_len = t_cache.dim(0) + t_curr.dim(0);
                    let b = t_cache.dim(1);
                    let h = t_cache.dim(2);
                    let d = t_cache.dim(3);
                    let c = c.reshape(&[total_len, b, h, d])?;

                    c.transpose_axes(&[1, 0, 2, 3])?
                };

                // Update cache
                cache.key_cache = k_new.clone();
                cache.value_cache = v_new.clone();
                k = k_new;
                v = v_new;
            } else {
                // First token(s)
                cache.key_cache = k.clone();
                cache.value_cache = v.clone();
            }
            // Update offset by new tokens count
            cache.offset += seq_len as usize;
        }

        // Transpose for attention: [B, num_heads, L, head_dim]

        // Transpose for attention: [B, num_heads, L, head_dim]
        q = q.transpose_axes(&[0, 2, 1, 3])?;
        k = k.transpose_axes(&[0, 2, 1, 3])?;
        v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Expand K and V for grouped query attention
        // Repeat each KV head num_kv_groups times
        if self.num_kv_groups > 1 {
            // K: [B, num_kv_heads, L, head_dim] -> [B, num_heads, L, head_dim]
            k = self.repeat_kv(k, self.num_kv_groups)?;
            v = self.repeat_kv(v, self.num_kv_groups)?;
        }

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scale_array = Array::from_f32(1.0 / scale);

        // scores = (Q @ K.T) / sqrt(head_dim)
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let mut scores = q.matmul(&k_t)?;
        scores = scores.multiply(&scale_array)?;

        // Apply causal mask
        if let Some(mask) = mask {
            // If using cache and generating one token, we extract the last row of the mask
            // But usually for generation we assume only relevant mask part is passed or broadcast handled
            // For now, assume mask matches sequence length (full sequence or new token against full history)
            scores = scores.add(mask)?;
        }

        // Softmax and multiply by V
        let attn_weights = mlx_rs::ops::softmax_axis(&scores, -1, false)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Transpose back: [B, num_heads, L, head_dim] -> [B, L, num_heads, head_dim]
        let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3])?;

        // Reshape: [B, L, num_heads, head_dim] -> [B, L, num_heads * head_dim]
        let attn_output = attn_output.reshape(&[batch_size, seq_len, -1])?;

        // Output projection
        let mut output = self.o_proj.forward(attn_output.clone())?;
        if let Some(lora) = &mut self.o_proj_lora {
            let lora_out = lora.forward(&attn_output)?;
            output = output.add(&lora_out)?;
        }
        Ok(output)
    }

    fn repeat_kv(&self, x: Array, n_rep: i32) -> Result<Array, Exception> {
        if n_rep == 1 {
            return Ok(x);
        }

        let (b, num_kv_heads, seq_len, head_dim) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));

        // Expand and reshape to repeat KV heads
        // [B, num_kv_heads, L, head_dim] -> [B, num_kv_heads, n_rep, L, head_dim]
        let x = x.reshape(&[b, num_kv_heads, 1, seq_len, head_dim])?;

        // Broadcast to [B, n_kv, n_rep, L, D]
        // This ensures that when we flatten n_kv * n_rep, we get [k0, k0, k1, k1...] (Grouped)
        // instead of [k0, k1, k0, k1...] (Interleaved/Tiled) which the previous concat logic produced.
        let shape = [b, num_kv_heads, n_rep, seq_len, head_dim];
        let x = mlx_rs::ops::broadcast_to(&x, &shape)?;

        // Reshape to [B, num_kv_heads * n_rep, L, head_dim]
        x.reshape(&[b, num_kv_heads * n_rep, seq_len, head_dim])
    }
}

/// Llama MLP with gated activation
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LlamaMLP {
    #[param]
    pub gate_proj: LinearLayer,
    #[param]
    pub up_proj: LinearLayer,
    #[param]
    pub down_proj: LinearLayer,
}

impl LlamaMLP {
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let mut gate_proj = LinearLayer::F16(Linear::new(config.hidden_size, config.intermediate_size)?);
        let mut up_proj = LinearLayer::F16(Linear::new(config.hidden_size, config.intermediate_size)?);
        let mut down_proj = LinearLayer::F16(Linear::new(config.intermediate_size, config.hidden_size)?);

        // Disable biases if configured (default for Llama 3)
        if !config.mlp_bias {
            if let LinearLayer::F16(l) = &mut gate_proj { *l.bias = None; }
            if let LinearLayer::F16(l) = &mut up_proj { *l.bias = None; }
            if let LinearLayer::F16(l) = &mut down_proj { *l.bias = None; }
        }

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn new_skeleton(_config: &LlamaConfig) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: LinearLayer::new_skeleton()?,
            up_proj: LinearLayer::new_skeleton()?,
            down_proj: LinearLayer::new_skeleton()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // gate = silu(gate_proj(x))
        let gate = self.gate_proj.forward(x.clone())?;
        let gate = mlx_rs::nn::silu(&gate)?;

        // up = up_proj(x)
        let up = self.up_proj.forward(x.clone())?;

        // output = down_proj(gate * up)
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(hidden)
    }
}

/// Single Llama decoder layer
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LlamaDecoderLayer {
    #[param]
    pub self_attn: LlamaAttention,
    #[param]
    pub mlp: LlamaMLP,
    #[param]
    pub input_layernorm: RmsNorm,
    #[param]
    pub post_attention_layernorm: RmsNorm,
}

impl LlamaDecoderLayer {
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let self_attn = LlamaAttention::new(config)?;
        let mlp = LlamaMLP::new(config)?;
        let input_layernorm = RmsNorm::new(config.hidden_size)?;
        let post_attention_layernorm = RmsNorm::new(config.hidden_size)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn new_skeleton(config: &LlamaConfig) -> Result<Self, Exception> {
        let self_attn = LlamaAttention::new_skeleton(config)?;
        let mlp = LlamaMLP::new_skeleton(config)?;
        // Norms are small (vector size), but we can make them 1-element too
        let input_layernorm = RmsNorm::new(1)?;
        let post_attention_layernorm = RmsNorm::new(1)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        // Pre-norm attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_output = self.self_attn.forward(&normed, mask)?;
        let x = x.add(&attn_output)?;

        // Pre-norm MLP with residual
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_output = self.mlp.forward(&normed)?;
        x.add(&mlp_output)
    }
}

/// Full Llama model (without lm_head)
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LlamaModel {
    pub config: LlamaConfig,
    #[param]
    pub embed_tokens: Embedding,
    #[param]
    pub layers: Vec<LlamaDecoderLayer>,
    #[param]
    pub norm: RmsNorm,
}

impl LlamaModel {
    pub fn new(config: LlamaConfig) -> Result<Self, Exception> {
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new(&config)?);
        }

        let norm = RmsNorm::new(config.hidden_size)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn new_skeleton(config: LlamaConfig) -> Result<Self, Exception> {
        // Embedding: 1x1
        let mut embed_tokens = Embedding::new(1, 1)?;
        *embed_tokens.weight = Array::from_slice(&[0.0f32], &[1, 1]);

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new_skeleton(&config)?);
        }

        let norm = RmsNorm::new(1)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn forward(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Create causal mask
        let seq_len = input_ids.dim(1);
        let mask = self.create_causal_mask(seq_len)?;

        // Pass through all decoder layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, Some(&mask))?;
        }

        // Final normalization
        self.norm.forward(&hidden_states)
    }

    fn create_causal_mask(&self, seq_len: i32) -> Result<Array, Exception> {
        // Create additive causal mask: 0 for allowed positions, -inf for masked
        let indices = mlx_rs::ops::arange::<_, f32>(0, seq_len, 1)?;
        let row = mlx_rs::ops::expand_dims(&indices, 0)?;
        let col = mlx_rs::ops::expand_dims(&indices, 1)?;

        // mask[i,j] = 1 if i < j (future positions), 0 otherwise
        // row is j, col is i. We want 1 where j > i.
        let mask = row.gt(&col)?;

        // Convert to f32 and multiply by large negative number
        let mask = mask.as_type::<f32>()?;
        let neg_inf = Array::from_f32(-1e9_f32);
        mask.multiply(&neg_inf)
    }
}

/// Frozen backbone - never participates in gradient computation
/// This prevents MLX from allocating gradient Arrays for frozen parameters
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LlamaBackbone {
    #[param]
    pub embed_tokens: Embedding,
    #[param]
    pub layers: Vec<LlamaDecoderLayer>,
    pub config: LlamaConfig,
}

impl LlamaBackbone {
    pub fn new(config: LlamaConfig) -> Result<Self, Exception> {
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new(&config)?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            config,
        })
    }

    pub fn new_skeleton(config: LlamaConfig) -> Result<Self, Exception> {
         let mut embed_tokens = Embedding::new(1, 1)?;
        *embed_tokens.weight = Array::from_slice(&[0.0f32], &[1, 1]);

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new_skeleton(&config)?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            config,
        })
    }

    /// Forward pass through frozen backbone (for use outside gradient graph)
    pub fn forward(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Create causal mask
        let seq_len = input_ids.dim(1);
        let mask = self.create_causal_mask(seq_len)?;

        // Pass through all decoder layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, Some(&mask))?;
        }

        Ok(hidden_states)
    }

    fn create_causal_mask(&self, seq_len: i32) -> Result<Array, Exception> {
        let indices = mlx_rs::ops::arange::<_, f32>(0, seq_len, 1)?;
        let row = mlx_rs::ops::expand_dims(&indices, 0)?;
        let col = mlx_rs::ops::expand_dims(&indices, 1)?;
        // mask[i,j] = 1 if i < j (future positions), 0 otherwise
        // row is j, col is i. We want 1 where j > i.
        let mask = row.gt(&col)?;
        let mask = mask.as_type::<f32>()?;
        let neg_inf = Array::from_f32(-1e9_f32);
        mask.multiply(&neg_inf)
    }

    pub fn setup_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.kv_cache = Some(KVCache::new());
        }
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.kv_cache = None;
        }
    }
}

/// Trainable head - only these parameters get gradients
/// This is the KEY to zero memory leaks - value_and_grad only sees these params
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct TrainableHead {
    #[param]
    pub norm: RmsNorm,
    #[param]
    pub lm_head: Linear,
}

impl TrainableHead {
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let norm = RmsNorm::new(config.hidden_size)?;
        let mut lm_head = Linear::new(config.hidden_size, config.vocab_size)?;

        // Always disable bias for Llama head to match checkpoints
        *lm_head.bias = None;

        Ok(Self { norm, lm_head })
    }

    pub fn new_skeleton(_config: &LlamaConfig) -> Result<Self, Exception> {
        let norm = RmsNorm::new(1)?;
        let mut lm_head = Linear::new(1, 1)?;
        *lm_head.weight = Array::from_slice(&[0.0f32], &[1, 1]);
        *lm_head.bias = None;

        Ok(Self { norm, lm_head })
    }

    /// Forward pass through trainable head (for use in gradient computation)
    pub fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        let normalized = self.norm.forward(hidden_states)?;
        self.lm_head.forward(&normalized)
    }
}

/// Llama model for causal language modeling with split architecture
/// Backbone is frozen, only head (or LoRA adapters) participate in gradients
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LlamaForCausalLM {
    #[param]
    pub backbone: LlamaBackbone,
    #[param]
    pub head: TrainableHead,
    // LoRA adapters will be added later
    pub lora_rank: usize,
}

impl LlamaForCausalLM {
    pub fn new(config: LlamaConfig) -> Result<Self, Exception> {
        let backbone = LlamaBackbone::new(config.clone())?;
        let head = TrainableHead::new(&config)?;

        Ok(Self {
            backbone,
            head,
            lora_rank: 0,
        })
    }

    pub fn new_skeleton(config: LlamaConfig) -> Result<Self, Exception> {
        let backbone = LlamaBackbone::new_skeleton(config.clone())?;
        let head = TrainableHead::new_skeleton(&config)?;

        Ok(Self {
            backbone,
            head,
            lora_rank: 0,
        })
    }

    /// Create a minimal placeholder model to free memory during reloads
    pub fn new_placeholder(mut config: LlamaConfig) -> Result<Self, Exception> {
        config.num_hidden_layers = 0;
        config.vocab_size = 128; // Smaller vocab for placeholder
        Self::new_skeleton(config)
    }

    pub fn forward(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        let hidden_states = self.backbone.forward(input_ids)?;
        self.head.forward(&hidden_states)
    }

    /// Forward through backbone only (returns hidden states before head)
    /// Use this outside gradient computation to prevent memory leaks
    pub fn forward_backbone(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.backbone.forward(input_ids)
    }

    /// Forward through head only (for use in gradient computation)
    pub fn forward_head(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        self.head.forward(hidden_states)
    }

    pub fn setup_cache(&mut self) {
        self.backbone.setup_cache();
    }

    pub fn clear_cache(&mut self) {
        self.backbone.clear_cache();
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.backbone.config
    }

    /// Generate text autoregressively from input token IDs
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs [batch_size, seq_len]
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (0.0 = greedy, >0.0 = sampling)
    ///
    /// # Returns
    /// Vector of generated token IDs (including input tokens)
    pub fn generate(
        &mut self,
        input_ids: &Array,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<i32>, Exception> {
        let batch_size = input_ids.dim(0);
        if batch_size != 1 {
            return Err(Exception::custom(
                "generate() only supports batch_size=1 currently",
            ));
        }

        // Initialize KV Cache
        self.setup_cache();

        // Convert input to vector
        let mut generated: Vec<i32> = input_ids.as_slice::<i32>().to_vec();
        let initial_len = generated.len();

        // 1. Prefill - process prompt
        // Forward full input to populate cache
        let logits = self.forward(input_ids)?;
        let logits = logits.as_type::<f32>()?;

        // Get logits for last token: [1, seq_len, vocab_size]
        let vocab_size = logits.dim(2);
        let seq_len = logits.dim(1);

        let logits_vec: Vec<f32> = logits.as_slice::<f32>().to_vec();
        let last_pos_start = ((seq_len - 1) * vocab_size) as usize;
        let last_pos_end = (seq_len * vocab_size) as usize;
        let last_logits_vec = logits_vec[last_pos_start..last_pos_end].to_vec();
        let mut last_logits = Array::from_slice::<f32>(&last_logits_vec, &[vocab_size]);

        // Sample first new token
        let mut next_token = if temperature < 1e-6 {
             let probs_vec: Vec<f32> = last_logits.as_slice::<f32>().to_vec();
                probs_vec
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i32)
                    .unwrap_or(0)
        } else {
            let scaled_logits = last_logits.divide(Array::from_f32(temperature))?;
            let probs = mlx_rs::ops::softmax_axis(&scaled_logits, -1, false)?;
            let probs_vec: Vec<f32> = probs.as_slice::<f32>().to_vec();
            sample_categorical(&probs_vec)
        };

        generated.push(next_token);

        // Check for EOS immediately after first token
        let is_eos = match &self.backbone.config.eos_token_id {
            Some(EosToken::Single(id)) => next_token == *id,
            Some(EosToken::Multiple(ids)) => ids.contains(&next_token),
            None => next_token == 2,
        };

        if !is_eos {
            // 2. Decode loop - generate one token at a time
            for _ in 1..max_new_tokens {
                // Prepare input: just the last token
                let input = Array::from_slice::<i32>(&[next_token], &[1, 1]);

                // Forward pass (uses cache offset implicitly)
                let logits = self.forward(&input)?;
                let logits = logits.as_type::<f32>()?;

                // Logits shape is [1, 1, vocab_size]
                // We just need the payload
                let logits_vec: Vec<f32> = logits.as_slice::<f32>().to_vec();
                last_logits = Array::from_slice::<f32>(&logits_vec, &[vocab_size]);

                // Sample next token
                next_token = if temperature < 1e-6 {
                    let probs_vec: Vec<f32> = last_logits.as_slice::<f32>().to_vec();
                    probs_vec
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as i32)
                        .unwrap_or(0)
                } else {
                    let scaled_logits = last_logits.divide(Array::from_f32(temperature))?;
                    let probs = mlx_rs::ops::softmax_axis(&scaled_logits, -1, false)?;
                    let probs_vec: Vec<f32> = probs.as_slice::<f32>().to_vec();
                    sample_categorical(&probs_vec)
                };

                generated.push(next_token);

                // Check for EOS
                let is_eos = match &self.backbone.config.eos_token_id {
                    Some(EosToken::Single(id)) => next_token == *id,
                    Some(EosToken::Multiple(ids)) => ids.contains(&next_token),
                    None => next_token == 2,
                };

                if is_eos {
                    break;
                }
            }
        }

        // Cleanup
        self.clear_cache();

        // Return only newly generated tokens (exclude input)
        Ok(generated[initial_len..].to_vec())
    }
}

/// Sample from categorical distribution
fn sample_categorical(probs: &[f32]) -> i32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let sample: f32 = rng.gen();

    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if sample < cumsum {
            return i as i32;
        }
    }

    // Fallback to last token
    (probs.len() - 1) as i32
}

/// Helper to load weights from safetensors into model
///
/// Loads pre-trained weights into a LlamaForCausalLM model.
/// This function maps safetensors weight names to model parameters.
pub fn load_weights_into_model(
    model: &mut LlamaForCausalLM,
    weights: HashMap<String, Array>,
) -> anyhow::Result<()> {
    println!("Loading {} weight tensors into model...", weights.len());

    let mut loaded_count = 0;
    let mut missing_keys: Vec<String> = Vec::new();
    let mut extra_keys: Vec<String> = Vec::new();
    let mut used_weight_keys = std::collections::HashSet::new();

    // Get mutable access to model parameters
    let mut parameters = model.parameters_mut().flatten();

    // Load weights from safetensors into model parameters
    // Handle name translation for split architecture:
    // - "model.layers.X" → "backbone.layers.X"
    // - "model.norm" → "head.norm"
    // - "lm_head" → "head.lm_head"
    // - "model.embed_tokens" → "backbone.embed_tokens"
    for (param_name, param) in parameters.iter_mut() {
        let param_name_str = param_name.to_string();

        // Try direct match first
        if let Some(weight_array) = weights.get(&param_name_str) {
            if weight_array.shape() == param.shape() {
                **param = weight_array.clone();
                let _ = param.eval();
                loaded_count += 1;
                used_weight_keys.insert(param_name_str);
                continue;
            }
        }

        // Try legacy name mapping for split architecture compatibility
        let legacy_name = if param_name_str.starts_with("backbone.") {
            // "backbone.layers.X" → "model.layers.X"
            // "backbone.embed_tokens" → "model.embed_tokens"
            param_name_str.replace("backbone.", "model.")
        } else if param_name_str.starts_with("head.norm") {
            // "head.norm.weight" → "model.norm.weight"
            // "head.norm" → "model.norm"
            param_name_str.replace("head.norm", "model.norm")
        } else if param_name_str.starts_with("head.lm_head") {
            // "head.lm_head.weight" → "lm_head.weight"
            // "head.lm_head" → "lm_head"
            param_name_str.replacen("head.", "", 1)
        } else {
            param_name_str.clone()
        };

        if let Some(weight_array) = weights.get(&legacy_name) {
            if weight_array.shape() == param.shape() {
                **param = weight_array.clone();
                let _ = param.eval();
                loaded_count += 1;
                used_weight_keys.insert(legacy_name);
                continue;
            } else {
                eprintln!(
                    "Warning: Shape mismatch for {} (legacy: {}): expected {:?}, got {:?}",
                    param_name_str,
                    legacy_name,
                    param.shape(),
                    weight_array.shape()
                );
            }
        }

        // Not found with either name
        missing_keys.push(param_name_str);
    }

    // Find extra keys in weights that don't match any model parameters
    for weight_key in weights.keys() {
        if !used_weight_keys.contains(weight_key) {
            extra_keys.push(weight_key.clone());
        }
    }

    println!(
        "Successfully loaded {} / {} weight tensors into model",
        loaded_count,
        parameters.len()
    );

    if !missing_keys.is_empty() {
        println!(
            "Missing keys (first 10): {:?}",
            &missing_keys[..missing_keys.len().min(10)]
        );
    }

    if !extra_keys.is_empty() {
        println!(
            "Extra keys in safetensors (first 10): {:?}",
            &extra_keys[..extra_keys.len().min(10)]
        );
    }

    if loaded_count == 0 {
        // Enhanced debugging: print sample parameter names and safetensors keys
        eprintln!("\nERROR: Parameter name mismatch detected!");
        eprintln!("No weights were successfully loaded into the model.");

        if weights.is_empty() {
            eprintln!("\nThe weights HashMap is empty!");
            eprintln!("This should have been caught by the caller - please use random initialization instead.");
        } else {
            let param_names: Vec<String> = parameters.keys().map(|k| k.to_string()).collect();
            let weight_keys: Vec<String> = weights.keys().cloned().collect();

            eprintln!("\nSample model parameter names (first 5):");
            for name in param_names.iter().take(5) {
                eprintln!("  - {}", name);
            }

            eprintln!("\nSample safetensors keys (first 5):");
            for key in weight_keys.iter().take(5) {
                eprintln!("  - {}", key);
            }
        }

        anyhow::bail!(
            "Failed to load any weights - parameter names may not match safetensors keys"
        );
    }

    Ok(())
}

/// Create a new LlamaForCausalLM model with pre-loaded weights
///
/// This is an alternative constructor that loads weights during model creation.
pub fn load_model_with_weights(
    config: LlamaConfig,
    weights: HashMap<String, Array>,
) -> anyhow::Result<LlamaForCausalLM> {
    // First create the model with random initialization
    let mut model = LlamaForCausalLM::new(config)?;

    // Then load the weights
    load_weights_into_model(&mut model, weights)?;

    Ok(model)
}
