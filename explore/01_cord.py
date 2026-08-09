# %%
from datasets import load_dataset

ds = load_dataset("naver-clova-ix/cord-v2")
print(ds)

# %%
import json

ex = ds["train"][0]
print(ex.keys())
print(ex["image"].size, ex["image"].mode)

# %%
ex["image"]

# %%
gt = json.loads(ex["ground_truth"])
print(json.dumps(gt, indent=2, ensure_ascii=False))

# %%
from collections import Counter

keys = Counter()
for i in range(100):
    g = json.loads(ds["train"][i]["ground_truth"])
    keys.update(g.get("gt_parse", {}).keys())

for k, n in keys.most_common():
    print(f"{n:3d}/100  {k}")

# %%
# one level deeper: what fields actually live inside menu / sub_total / total?
sub_keys = {"menu": Counter(), "sub_total": Counter(), "total": Counter()}

for i in range(100):
    g = json.loads(ds["train"][i]["ground_truth"]).get("gt_parse", {})
    for cat in sub_keys:
        val = g.get(cat)
        if val is None:
            continue
        items = val if isinstance(val, list) else [val]
        for item in items:
            if isinstance(item, dict):
                sub_keys[cat].update(item.keys())

for cat, counter in sub_keys.items():
    print(f"--- {cat} ---")
    for k, n in counter.most_common():
        print(f"  {n:3d}  {k}")
# %%
