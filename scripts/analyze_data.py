import json
import statistics


def analyze(file_path):
    print(f"Analyzing {file_path}...")
    auth_weights = []
    prov_entropies = []

    try:
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if "auth_weight" in data:
                        auth_weights.append(data["auth_weight"])
                    if "prov_entropy" in data:
                        prov_entropies.append(data["prov_entropy"])
                except:
                    pass
                if i % 10000 == 0 and i > 0:
                    print(f"Processed {i} lines...")
    except FileNotFoundError:
        print("File not found.")
        return

    total = len(auth_weights)
    if total == 0:
        print("No valid data found.")
        return

    print(f"\nTotal Examples: {total}")

    # Authority Weight Stats
    avg_auth = statistics.mean(auth_weights)
    print("\nAuthority Weight (Lower is better/Primary):")
    print(f"  Mean: {avg_auth:.4f}")
    print(f"  Min:  {min(auth_weights):.4f}")
    print(f"  Max:  {max(auth_weights):.4f}")

    # Buckets
    primary = sum(1 for x in auth_weights if x < 0.3)
    mid = sum(1 for x in auth_weights if 0.3 <= x <= 0.8)
    consensus = sum(1 for x in auth_weights if x > 0.8)

    print(f"  Primary Sources (<0.3):    {primary} ({primary / total * 100:.1f}%)")
    print(f"  Mixed/Average (0.3-0.8):   {mid} ({mid / total * 100:.1f}%)")
    print(f"  Modern Consensus (>0.8):   {consensus} ({consensus / total * 100:.1f}%)")

    # Entropy Stats
    avg_ent = statistics.mean(prov_entropies)
    print("\nProvenance Entropy (Higher is better/Primary):")
    print(f"  Mean: {avg_ent:.4f}")
    print(f"  Max:  {max(prov_entropies):.4f}")

    high_ent = sum(1 for x in prov_entropies if x > 5.0)
    print(f"  High Entropy (>5.0 bits):  {high_ent} ({high_ent / total * 100:.1f}%)")


if __name__ == "__main__":
    analyze("data/train.jsonl")
