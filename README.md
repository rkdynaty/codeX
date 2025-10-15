# codeX

Project to try out OpenAI codeX

## Stock Analysis CLI

This repository includes a command-line interface (CLI) at `scripts/analyze_stock.py` that downloads and analyzes historical stock data from Yahoo Finance using only the Python standard library.

### Requirements

- Python 3.9 or newer (the standard library modules used are available on Python 3.9+)
- Internet access is optional; you can provide a local Yahoo Finance-formatted CSV file when running offline.

### Usage

Display the available options:

```bash
python scripts/analyze_stock.py --help
```

Download the latest year of Apple (AAPL) data, compute the default moving averages (20, 50, 200 days), and print the summary report:

```bash
python scripts/analyze_stock.py AAPL
```

Analyze an offline dataset using the bundled sample file (`data/sample_aapl.csv`) and customize the moving-average windows:

```bash
python scripts/analyze_stock.py AAPL --data-file data/sample_aapl.csv --moving-averages 5 10 20
```

Export the enriched dataset to a CSV file after computing indicators:

```bash
python scripts/analyze_stock.py TSLA --start 2023-01-01 --export data/tsla_analysis.csv
```

### Testing the Script

To verify the script locally, run the following commands:

1. Show the CLI help text to ensure the script runs:
   ```bash
   python scripts/analyze_stock.py --help
   ```
2. Execute the CLI against the bundled sample data to confirm offline analysis works:
   ```bash
   python scripts/analyze_stock.py AAPL --data-file data/sample_aapl.csv --moving-averages 5 10
   ```
3. (Optional) Run the CLI against live Yahoo Finance data if you have network access:
   ```bash
   python scripts/analyze_stock.py MSFT --start 2023-01-01
   ```

These commands should complete without errors, producing console summaries and, when `--export` is provided, a CSV file containing the computed indicators.
