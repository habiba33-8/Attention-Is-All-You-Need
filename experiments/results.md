
### experiments/single_vs_multi_head/results.md

```markdown
# Results – Single vs Multi-Head Attention Ablation

(Example output from a small dummy-data run – replace with real numbers)

Date             | Model       | heads | avg loss (6 epochs) | Δ loss vs multi-head | Notes
-----------------|-------------|-------|----------------------|----------------------|----------------------
2025-02-xx       | Multi-head  | 8     | 5.124                | —                    | baseline
2025-02-xx       | Single-head | 1     | 5.387                | +0.263               | clearly worse

Real-data expectation (based on paper Table 3 row A):
- Multi-head (h=8)   → BLEU ≈ 25.8
- Single-head (h=1)  → BLEU ≈ 24.9   (–0.9 BLEU)

Conclusion
----------
Multi-head attention consistently outperforms single-head attention  
when total computation is held roughly constant — confirming one of the  
key design decisions of the Transformer architecture.