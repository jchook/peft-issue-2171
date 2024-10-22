# peft-issue-2171

See issue [#2171](huggingface/peft issue #2171 - non-deterministic, erratic results from loading LoRA adapters) for more info.

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

