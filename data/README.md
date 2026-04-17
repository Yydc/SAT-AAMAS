# Datasets

`scripts/prepare_data.py` writes the JSONL files used by the training and
evaluation entry points into this directory. Pick the dataset you need:

| Dataset  | Command                                      | Output                              |
| -------- | -------------------------------------------- | ----------------------------------- |
| `demo`   | `python scripts/prepare_data.py --dataset demo`   | `data/demo/{train,test}.jsonl`        |
| `aime24` | `python scripts/prepare_data.py --dataset aime24` | `data/aime24/test.jsonl`              |
| `aime25` | `python scripts/prepare_data.py --dataset aime25` | `data/aime25/test.jsonl`              |
| `math500`| `python scripts/prepare_data.py --dataset math500`| `data/math500/{train,test}.jsonl`     |
| `dapo`   | `python scripts/prepare_data.py --dataset dapo`   | `data/dapo/train.jsonl`               |

The non-demo datasets pull from Hugging Face Datasets, so install
`pip install datasets` before running them.

## JSONL schema

* **Training (`train.jsonl`)**
  ```json
  {"prompt": "Problem ...", "chosen": "Reference answer", "rejected": "Optional"}
  ```
* **Evaluation (`test.jsonl`)**
  ```json
  {"problem": "Problem ...", "answer": "Reference answer"}
  ```

Both schemas are accepted by `RealMultiAgentController`; legacy keys
(`prompt`/`answer`) are mapped automatically.
