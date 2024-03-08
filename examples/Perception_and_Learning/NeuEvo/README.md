# NeuEvo

TODO: Add description

## Train

Set the path of the dataset in `main.py`:

```python
loader_train, loader_eval, mixup_active, mixup_fn = eval('get_%s_data' % args.dataset)(
    ...,
    root="./data", # change root parameter to your own dataset path
)
```

Select a SNN architecture as showed in `braincog.model_zoo.NeuEvo.genotypes`, defaults to `"mlp1"`:

```python
from braincog.model_zoo.NeuEvo import genotypes
...
parser.add_argument('--arch', default='mlp1', type=str) # change target SNN architecture
```

Execute:

```bash
python ./main.py
```