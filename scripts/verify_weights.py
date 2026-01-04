import sys

import numpy as np
import safetensors.numpy


def verify_weights(path):
    print(f"Checking {path}...")
    try:
        with safetensors.numpy.safe_open(path, framework="np") as f:
            tensors = f.keys()
            print(f"Found {len(tensors)} tensors.")

            non_zero_keys = []

            for key in tensors:
                tensor = f.get_tensor(key)
                if np.any(tensor != 0):
                    non_zero_keys.append(key)

            print(f"Total parameters: {sum(f.get_tensor(k).size for k in tensors)}")
            print(f"Tensors with non-zero values: {len(non_zero_keys)}/{len(tensors)}")

            print("\nNon-zero tensors:")
            for k in non_zero_keys:
                print(f"  - {k}")

            if len(non_zero_keys) == 0:
                print("FAILURE: All weights are zero!")
                sys.exit(1)
            else:
                print("SUCCESS: Weights appear valid (technically).")
                sys.exit(0)

    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_weights.py <path_to_safetensors>")
        sys.exit(1)
    verify_weights(sys.argv[1])
