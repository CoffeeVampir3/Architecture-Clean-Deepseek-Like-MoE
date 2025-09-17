A clean "baseline" moe transformer version to compared against Qwen3 next architecture. Architcturally:

- Deep seek style MoE (auxillary loss free routing)
- Zero Centered RMS Norm /w Weight Decay

Using uv:
```
uv sync
```

Train (trains on wikitext-2-v1 for 10 epochs by default) 
```
uv run python main.py
```

Infer (hard coded to use checkpoint 10):
```
uv run python basic_inf.py
```


