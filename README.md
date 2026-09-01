# `pthin`: post-selecton inference with randomized p-values

In settings where hypotheses are selected using p-values (e.g., selecting the sample with the lowest p-value, or any with p-value below $0.05$), using the same data for inference results in tests with inflated type 1 error rates, confidence intervals with below nominal coverage, and estimators that can demonstrate severe bias. This package implements methods to (i) construct randomized p-values, and (ii) provide valid inference on parameters selected using these randomized p-values.

## Installation

```bash
pip install git+https://github.com/rflperry/pthin.git
```

## Usage

`pthin` splits each p-value in `p` into a pair `(p1, p2)` such that if `p` is 
uniformly distributed (e.g., the null is true), then `p1` and `p2` are (i)
both uniformly distributed and (ii) independent of each other.
This allows one to be used for selection and the other for inference.

```python
import numpy as np
from scipy.stats import norm
from pthin import pthin, pcarve_threshold, pcarve_ci, pcarve_estimate

# p-values and significance level
p = np.array([0.01, 0.1, 0.5, 0.9])
alpha = 0.05

# Thin the p-values
epsilon = 0.5
p1, p2 = pthin(p, epsilon=epsilon, rng=np.random.default_rng(0))

# File drawer selection
sel = np.where(p1 <= alpha)[0]
cutoff = alpha

# Winner's curse selection
sel = np.argmin(p1)
cutoff = np.sort(p1)[1]

# p-thinning test
p2[sel] <= alpha
```

When the selection event can be characterized as the event that the selected p-value lies
in an interval (e.g., the winner's curse or the file drawer problem), `pthin` further provides confidence intervals, point estimators, and more powerful tests (see `pcarve_ci` and `pcarve_threshold` in
[`pthin/inference.py`](pthin/inference.py), and `pcarve_estimate` in
[`pthin/estimate.py`](pthin/estimate.py)).

```python
# p-carving test
p[sel] <= pcarve_threshold(
   alpha=alpha, b=cutoff,
   epsilon=epsilon,
)

# p-carving confidence interval (assuming p-values from a Z-test)
pcarve_ci(
   p[sel], b=cutoff,
   epsilon=epsilon, alpha=alpha,
)

# p-carving estimation (assuming p-values from a Z-test)
pcarve_estimate(
   p[sel], b=cutoff, epsilon=epsilon
)
```

## Experimental reproducibility

All simulation studies and figures can be replicated in the notebooks found in `experiments/`.

To process the UK Biobank summary statistics in `experiments/ukbiobank.ipynb`, you must download the raw VCF files manually.

1. **Download the Data**: Navigate to the following datasets and click **"VCF Download Links"** to download the compressed `.vcf.gz` files:
   - **BMI:** [ukb-b-19953](https://opengwas.io/datasets/ukb-b-19953)
   - **Height:** [ukb-b-10787](https://opengwas.io/datasets/ukb-b-10787)
   - **Type 2 Diabetes:** [ebi-a-GCST006867](https://opengwas.io/datasets/ebi-a-GCST006867)

2. **Stage the Files**: Place the three downloaded `.vcf.gz` files directly into the `data/` directory.

3. **Format for Analysis**: Run the formatting script to parse the VCFs and extract the SNPs, $p$-values, and effect signs:
   ```bash
   python data/get_biobank_summaries.py
    ```

## Development

With [uv](https://docs.astral.sh/uv/) (recommended — uses the checked-in
`uv.lock` for a reproducible environment):

```bash
git clone https://github.com/rflperry/pthin.git
cd pthin
uv sync --group dev  # add --group notebooks too if you want to run experiments/*.ipynb
uv run pytest  # run the test suite
```

Or with plain `pip`:

```bash
git clone https://github.com/rflperry/pthin.git
cd pthin
pip install -e .
pip install --group dev  # add --group notebooks too if you want to run experiments/*.ipynb
pytest
```