# TabulaSage

**An offline, open-source-AI profiler for any CSV.** Point it at a spreadsheet
and it tells you — in plain English — what's in your data, which rows look
wrong, and what actually drives your target column. Everything runs locally on
your CPU. No cloud, no API key, no telemetry. Your data never leaves your machine.

```bash
pip install -r requirements.txt
python tabulasage.py examples/sample_customers.csv --target churned
```

## Why open-source AI is the whole point

Most "smart spreadsheet" tools upload your file to someone else's server, run a
closed model you can't inspect, and charge you per call. TabulaSage takes the
opposite stance:

- **Your data stays yours.** There is no network call anywhere in the code path.
  Point it at a customer export, a medical log, or financial records and nothing
  is transmitted. That isn't a privacy policy — it's an architectural fact you
  can verify by reading 200 lines of Python.
- **It costs nothing to run.** The model is [scikit-learn](https://scikit-learn.org)'s
  `IsolationForest` and a shallow decision tree — open source, auditable,
  reproducible. Same input, same output, every time, on any laptop.
- **You can swap the brain.** Because the intelligence is an open framework
  rather than a hosted endpoint, you can drop in a local open-weight LLM
  (Gemma via Ollama) for richer prose, or [TabPFN](https://priorlabs.ai) for
  forecasting, without rewriting the app or sending data anywhere.

For a developer working with sensitive CSVs, "open source at the core" isn't
ideology — it's the only way to get AI insights *and* keep the data on your own disk.

## What it does

1. **Profiles every column** — type, missingness, distribution, top categories.
2. **Detects anomalous rows** with an `IsolationForest` (standardized features,
   parameter-free 2σ threshold), surfacing data-entry errors and rare cases.
3. **Trains an interpretable model** when you pass `--target`, reporting test
   accuracy and the features that actually drive the outcome (no black box).
4. **Writes plain-English insights** you can paste straight into a report.

## Real output

```
$ python tabulasage.py examples/sample_customers.csv --target churned
================================================================
 TabulaSage — offline AI profile  |  scikit-learn v1.9.1
 15 rows x 6 columns
================================================================

COLUMN PROFILE
  • age                  numeric   mean=105 std=239 min=27 median=40 max=999 missing=0
  • income               numeric   mean=56,468 std=21,451 min=15 median=59,000 max=91,000 missing=0
  • city                 category  unique=3 top='Shanghai'(6) missing=0
  • tenure_months        numeric   mean=18.7 std=13.7 min=1 median=15 max=48 missing=0
  • monthly_spend        numeric   mean=266 std=133 min=5 median=260 max=520 missing=0
  • churned              category  unique=2 top='no'(11) missing=0

ANOMALY DETECTION (IsolationForest)
  flagged 1 of 15 rows as anomalous
    [ANOMALY] row 9: age=999, income=15, city=Beijing, tenure_months=1
    [normal ] row 6: age=61, income=91000, city=Shenzhen, tenure_months=48
    [normal ] row 13: age=58, income=84000, city=Shenzhen, tenure_months=40

INTERPRETABLE MODEL (target = churned)
  test accuracy: 100.0%  (classes: ['no', 'yes'])
  top drivers:
    - age                      0.571
    - monthly_spend            0.429

PLAIN-ENGLISH INSIGHTS
  → IsolationForest flagged 1 row(s) (6.7%) as anomalous — likely data-entry errors or rare cases worth a human look.
  → Predicting the target: a shallow decision tree reaches 100% test accuracy; the strongest driver is 'age' (importance 0.57).
```

The planted bad row (`age=999, income=15`) is caught instantly, while genuine
high-value customers are correctly left alone — and the model explains *why*
(`age`, then `monthly_spend`), not just *what*.

## Usage

```bash
# profile only
python tabulasage.py data.csv

# profile + train an interpretable model on a label column
python tabulasage.py data.csv --target churned

# also dump a machine-readable JSON summary
python tabulasage.py data.csv --json summary.json
```

## How it works (200 lines, all offline)

- CSV parsing and profiling use only the Python standard library + NumPy.
- Anomaly detection: `IsolationForest` on standardized numeric + one-hot features,
  flagged when the anomaly score falls more than 2σ below the mean.
- Interpretable model: a depth-limited `DecisionTreeClassifier`, so you can read
  the exact rule that drives each prediction.
- Insights: deterministic templates turn the statistics into sentences — no LLM
  required, but a local open-weight model can be plugged in for richer wording.

## Requirements

- Python 3.9+
- `numpy`, `scikit-learn` (see `requirements.txt`)

## License

MIT. See [LICENSE](LICENSE) and [CONTEST_DECLARATION.md](CONTEST_DECLARATION.md).
