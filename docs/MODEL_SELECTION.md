# Model Selection Guide

This document helps you choose the right EST model variant for your use case.

## Model Comparison Table

| Aspect | Reusable EST modules | EST+Equiformer | EST+EquiformerV2 |
|--------|----------------------|----------------|------------------|
| **For** | Adding EST to a custom backbone | Small-to-medium molecules | Large catalytic/material systems |
| **Datasets** | Backbone dependent | QM9, OC20 IS2RE | OC20 S2EF, MPtrj |

---

## Detailed Comparison

### Reusable EST modules

**Advantages:**
- Imports directly from `EST_nets`
- Adds `NodeEST` or `EdgeEST` without replacing the surrounding model
- Keeps the original exactly equivariant path as a parallel branch
- Supports custom spherical sampling resolution and expert counts

---

### EST+Equiformer

**Advantages:**
- Fastest to set up
- Clear, straightforward training scripts
- Good baseline performance
- Flexible hyperparameter control via command-line args
---

### EST+EquiformerV2

**Advantages:**
- Newest, most advanced Equiformer architecture
- Optimized for large catalytic systems
- Better scalability and performance
- Integrated with FairChem production framework

---
