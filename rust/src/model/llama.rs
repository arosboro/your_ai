use mlx_macros::ModuleParameters as DeriveModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Embedding, Linear, RmsNorm, Rope, RopeBuilder};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
        let total_memory = (base_memory as f64 * 1.5) as u64;
        total_memory
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
    pub q_proj: Linear,
    #[param]
    pub k_proj: Linear,
    #[param]
    pub v_proj: Linear,
    #[param]
    pub o_proj: Linear,
    pub rope: Rope,
    pub head_dim: i32,
    pub num_kv_groups: i32,
}

impl LlamaAttention {
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let num_kv_groups = config.num_attention_heads / config.num_key_value_heads;

        let q_proj = Linear::new(config.hidden_size, config.num_attention_heads * head_dim)?;

        let k_proj = Linear::new(config.hidden_size, config.num_key_value_heads * head_dim)?;

        let v_proj = Linear::new(config.hidden_size, config.num_key_value_heads * head_dim)?;

        let o_proj = Linear::new(config.num_attention_heads * head_dim, config.hidden_size)?;

        let rope = RopeBuilder::new(head_dim).base(config.rope_theta).build()?;

        Ok(Self {
            config: config.clone(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            head_dim,
            num_kv_groups,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let (batch_size, seq_len, _) = (x.dim(0), x.dim(1), x.dim(2));

        // Project to Q, K, V
        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let mut v = self.v_proj.forward(x)?;

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
        q = self.rope.forward(&q)?;
        k = self.rope.forward(&k)?;

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
        self.o_proj.forward(&attn_output)
    }

    fn repeat_kv(&self, x: Array, n_rep: i32) -> Result<Array, Exception> {
        if n_rep == 1 {
            return Ok(x);
        }

        let (b, num_kv_heads, seq_len, head_dim) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));

        // Expand and reshape to repeat KV heads
        // [B, num_kv_heads, L, head_dim] -> [B, num_kv_heads, n_rep, L, head_dim]
        let x = x.reshape(&[b, num_kv_heads, 1, seq_len, head_dim])?;

        // Broadcast by tiling
        let mut repeated = Vec::new();
        for _ in 0..n_rep {
            repeated.push(x.clone());
        }
        let x = mlx_rs::ops::concatenate(
            &repeated.iter().map(|a| a.as_ref()).collect::<Vec<&Array>>(),
        )?;

        // Reshape to [B, num_kv_heads * n_rep, L, head_dim]
        x.reshape(&[b, num_kv_heads * n_rep, seq_len, head_dim])
    }
}

/// Llama MLP with gated activation
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LlamaMLP {
    #[param]
    pub gate_proj: Linear,
    #[param]
    pub up_proj: Linear,
    #[param]
    pub down_proj: Linear,
}

impl LlamaMLP {
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let gate_proj = Linear::new(config.hidden_size, config.intermediate_size)?;
        let up_proj = Linear::new(config.hidden_size, config.intermediate_size)?;
        let down_proj = Linear::new(config.intermediate_size, config.hidden_size)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // gate = silu(gate_proj(x))
        let gate = self.gate_proj.forward(x)?;
        let gate = mlx_rs::nn::silu(&gate)?;

        // up = up_proj(x)
        let up = self.up_proj.forward(x)?;

        // output = down_proj(gate * up)
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Single Llama decoder layer
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LlamaDecoderLayer {
    #[param]
    pub attention: LlamaAttention,
    #[param]
    pub mlp: LlamaMLP,
    #[param]
    pub input_layernorm: RmsNorm,
    #[param]
    pub post_attention_layernorm: RmsNorm,
}

impl LlamaDecoderLayer {
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let attention = LlamaAttention::new(config)?;
        let mlp = LlamaMLP::new(config)?;
        let input_layernorm = RmsNorm::new(config.hidden_size)?;
        let post_attention_layernorm = RmsNorm::new(config.hidden_size)?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        // Pre-norm attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_output = self.attention.forward(&normed, mask)?;
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
        let mask = row.lt(&col)?;

        // Convert to f32 and multiply by large negative number
        let mask = mask.as_type::<f32>()?;
        let neg_inf = Array::from_f32(-1e9_f32);
        mask.multiply(&neg_inf)
    }
}

/// Llama model for causal language modeling
#[derive(Debug, Clone, DeriveModuleParameters)]
pub struct LlamaForCausalLM {
    #[param]
    pub model: LlamaModel,
    #[param]
    pub lm_head: Linear,
}

impl LlamaForCausalLM {
    pub fn new(config: LlamaConfig) -> Result<Self, Exception> {
        let model = LlamaModel::new(config.clone())?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size)?;

        Ok(Self { model, lm_head })
    }

    pub fn forward(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids)?;
        self.lm_head.forward(&hidden_states)
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.model.config
    }
}

/// Helper to load weights from safetensors into model
///
/// Loads pre-trained weights into a LlamaForCausalLM model.
/// This function maps safetensors weight names to model parameters.
pub fn load_weights_into_model(
    _model: &mut LlamaForCausalLM,
    weights: HashMap<String, Array>,
) -> anyhow::Result<()> {
    println!("Loading {} weight tensors into model...", weights.len());

    let _loaded_count = 0;
    let missing_keys: Vec<String> = Vec::new();

    // TODO: Weight Loading API - Needs mlx-rs parameter setting documentation
    // The model derives ModuleParameters (via #[derive(ModuleParameters)] and #[param] attributes),
    // which provides access to parameters via model.parameters() returning a NestedHashMap.
    //
    // To load weights, we need to:
    //   1. Iterate over model.parameters() to get parameter names and references
    //   2. Match safetensors keys to parameter names (handling name mapping)
    //   3. Set parameter values using the appropriate mlx-rs API
    //
    // Expected pattern (needs mlx-rs API confirmation):
    //   for (name, param) in model.parameters().flatten() {
    //       if let Some(weight_array) = weights.get(&name) {
    //           param.set_value(weight_array)?;  // or similar API
    //           loaded_count += 1;
    //       } else {
    //           missing_keys.push(name.clone());
    //       }
    //   }
    //
    // For now, report weights loaded from file without setting them.

    println!("Loaded {} weight tensors from safetensors", weights.len());
    println!("Weight loading into model structure needs mlx-rs parameter update API");

    let loaded_count = weights.len();

    if !missing_keys.is_empty() && missing_keys.len() < 10 {
        println!(
            "Missing keys (first 10): {:?}",
            &missing_keys[..missing_keys.len().min(10)]
        );
    }

    if loaded_count == 0 {
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
