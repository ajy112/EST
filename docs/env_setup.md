# Environment Setup

This document provides detailed environment setup instructions for different EST model variants.

## EST+Equiformer (QM9, OC20 IS2RE)

### Step 1: Create Conda Environment

```bash
conda env create -f env/env_est.yml
```

This creates a new environment called `est`.

### Step 2: Activate Environment

```bash
export PYTHONNOUSERSITE=True    # Prevent using packages from base environment
conda activate est
```

### Step 3: Install OC20 Models (Optional - Only for OC20 Training)

If you plan to train on OC20 datasets, install the `ocpmodels` package:

```bash
git clone https://github.com/FAIR-Chem/fairchem.git
cd fairchem
git checkout d2aaaeb67687785a99f869b4568e59cbbfab7acb
pip install -e .
```

**Note:** The version of `ocpmodels` used here is `0.0.3`. This corresponds to the FairChem commit [d2aaaeb67687785a99f869b4568e59cbbfab7acb](https://github.com/FAIR-Chem/fairchem/tree/d2aaaeb67687785a99f869b4568e59cbbfab7acb).

### Dependencies

All package versions are specified in [`env/env_est.yml`](../env/env_est.yml).

---

## EST+EquiformerV2 (OC20 S2EF, MPtrj)

EST+EquiformerV2 is deployed as part of the FairChem-1.10.0 framework.

### Setup Steps:

1. **Clone FairChem repository:**
   ```bash
   git clone https://github.com/FAIR-Chem/fairchem.git
   cd fairchem
   git checkout 977a803
   ```

2. **Integrate EST+EquiformerV2 models:**
   - Copy or symlink models from `equiformerv2-EST/` to the FairChem models directory

3. **Follow FairChem's environment setup:**
   ```bash
   pip install -e .
   ```

4. **Configure and train:**
   - Use EST+EquiformerV2 model config within FairChem's training framework

