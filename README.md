# peft-issue-2171

See issue [huggingface/peft#2171](https://github.com/huggingface/peft/issues/2171) for more info.

To replicate the bug, run the following (e.g. in a venv):

```sh
# Install required python packages
pip install -r requirements.txt

# Train a LoRA for two epochs
python train.py

# Run test.py 100 times on epoch 1
./test.sh

# Show the results
cat results.txt
```

See that the [results](./results.txt) are erratic and inconsistent:

```
...
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.265}
{'accuracy': 0.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.12}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 0.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 1.0}
{'accuracy': 0.68}
{'accuracy': 1.0}
```

