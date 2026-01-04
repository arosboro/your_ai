import argparse
import gc
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# Emulate 8B model size (approx)
# 8B params in float16 = 16GB
# We'll create a dummy model with multiple large layers to simulate this state
class Dummy8BModel(nn.Module):
    def __init__(self, layer_size=4096, num_layers=32):
        super().__init__()
        # 4096 * 4096 * 32 layers ~= 500M params.
        # Real 8B is much larger, let's bump it up.
        # Llama 3 8B: 32 layers, 4096 hidden, 14336 intermediate
        # Let's simulate ~4GB of params to be safe on local runs,
        # or scalable if we want to crash it.
        # User has 96GB, so let's try to allocate ~8GB model (4B params fp16)

        self.layers = []
        for _ in range(16):  # 16 layers of 4096*4096
            self.layers.append(nn.Linear(4096, 4096, bias=False))
        self.layers = list(self.layers)  # Register parameters

    def __call__(self, x):
        for l in self.layers:
            # Simple residual connection to keep gradient graph alive
            x = x + l(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="MLX Memory Verification Script")
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps to run"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    print("Initializing Dummy Model (~4GB params)...")
    model = Dummy8BModel()
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=1e-5)

    # model is captured from outer scope and updated in-place by nn.value_and_grad wrapper
    def loss_fn(model, X, y):
        logits = model(X)
        return mx.mean((logits - y) ** 2)

    # Transform OUTSIDE the loop/compiled function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Compile removed to simplify debugging and avoid capture issues
    # @mx.compile
    def step_fn(model, X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss

    print(f"Starting training loop for {args.steps} steps...")

    # Baseline memory
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()

    gc.collect()
    time.sleep(1)  # Let system settle

    start_time = time.time()

    for i in range(args.steps):
        # generate dummy batch
        X = mx.random.normal((args.batch_size, 4096)).astype(mx.float16)
        y = mx.random.normal((args.batch_size, 4096)).astype(mx.float16)

        # Eval inputs to force allocation
        mx.eval(X, y)

        # Step
        loss = step_fn(model, X, y)

        # CRITICAL: Force eval to ensure computation happens and graph is freed
        mx.eval(loss)

        # Periodic cleanup (mimicking mistral-finetune or just good hygiene)
        if i % 10 == 0:
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()  # Fallback

        if i % 10 == 0:
            active_mem = mx.metal.get_active_memory() / 1024**3
            peak_mem = mx.metal.get_peak_memory() / 1024**3
            cache_mem = mx.metal.get_cache_memory() / 1024**3
            print(
                f"Step {i:04d} | Loss: {loss.item():.4f} | Active: {active_mem:.2f}GB | Peak: {peak_mem:.2f}GB | Cache: {cache_mem:.2f}GB"
            )

    print("Verification complete.")


if __name__ == "__main__":
    main()
