### Processing the data

---
Bert accepts a fixed length of 512 tokens per sample.  It also requires a specialized tokenizer to prepare data for modeling.  To process the data simply run:

```shell script
python process_data.py --load_dir load_dir --save_dir save_dir
 
# load_dir = /path/to/raw/data
# save_dir = /path/to/save/processed/data
```

Optional:

```shell script
# filter out data from certain subredits by adding --filter
# toggle between expert/crowd data using --expert
# toggle between train or test data --test

# e.g.

python process_data.py --load_dir load_dir --save_dir save_dir --filter --expert --test
```


### Training a deep model

---
Initialize a model:

```python
from deep_models.model import Bert4Clf
model = Bert4Clf(use_cuda=True, joint_training=True)
```

The model consists of a linear classifier on top of BERT. Deep ontextualized featrures are first extracted from BERT and fed into the classifier.

Options:
```python
# use when a GPU is available
use_cuda=True

# jointly train BERT and the classifier
joint_training=True

# when this option is True, the learning rates will be
# set to 1e-5 and 1e-3 for BERT and the classifier respectively.
# this is to prevent catastrophic forgetting
```

For more details, please see the included notebooks.

### Validating a model

---
Load weights from a trained model:

```python
import torch
from deep_models.model import Bert4Clf

# init a new instance of the model
model = Bert4Clf(use_cuda=True, joint_training=True)

# load weights into the model
state_dict = torch.load('./model_filtered_3epochs.bin')
model.load_state_dict(state_dict)
```

We provide two set of weights:
- Model trained on the entire dataset: [model_3epochs.bin](https://drive.google.com/open?id=16y2MnxXxS6cz2igLBt0EVE0-G0MvT6EO)
- Model trained on filtered data: [model_filtered_3epochs.bin](https://drive.google.com/open?id=1vyQYMLy1BxvmJZtlk2mJmM0qBmFg9Q6f)



Validate the model using:

```python
user_true_pred_lbls = model.validate_model(
    test_data, batch_size=4)
```

Finally, calculate and print various performance scores:

```python
from deep_models.utils import calculate_metrics
calculate_metrics(user_true_pred_lbls, labels)
```

### Visualizing attention

---
Please see the included notebook for usage.


### Acknowledgments

---
We incorporated code from the following repos:
- https://github.com/huggingface/transformers
- https://github.com/jessevig/bertviz
