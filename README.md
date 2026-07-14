# p-value Thinning

Inference after selection on randomized p-values.

## Installation

```bash
pip install git+https://github.com/rflperry/pthin.git
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

## Usage

```python
import numpy as np
from pthin import pthin

p = np.array([0.01, 0.2, 0.5, 0.9])
p1, p2 = pthin(p, epsilon=0.5, rng=np.random.default_rng(0))
```

`pthin` splits each p-value in `p` into a pair `(p1, p2)` that, under the
null hypothesis, are each themselves valid (uniformly distributed) p-values
and approximately independent of one another. This allows one to be used to
select a hypothesis or threshold and the other for inference, without the
double-dipping that would come from reusing `p` for both. See the
[`pthin.randomize.pthin`](pthin/randomize.py) docstring for details.

`pthin` also provides post-selection inference built on top of this
construction (`pcarve_ci`, `pcarve_threshold`, `pcarve_estimate`, in
[`pthin/inference.py`](pthin/inference.py) and
[`pthin/estimate.py`](pthin/estimate.py)), plus the classic
truncated-Gaussian conditional-selective-inference baseline
(`truncgauss_pvalue`, `truncgauss_ci`, `truncgauss_estimate`) — see each
function's docstring for details.
