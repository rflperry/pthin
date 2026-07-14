# pval-thinning

Randomized thinning of p-values for independent post-selection inference.

## Installation

```bash
pip install git+https://github.com/<your-org>/<your-repo>.git
```

Or, for local development with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
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

## Tests

```bash
uv run pytest
```
