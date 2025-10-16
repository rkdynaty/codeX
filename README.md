# CodeX Stock Analysis CLI

A lightweight command-line tool for downloading, enriching, and summarising historical stock prices using only the Python standard library. The CLI can operate online via Yahoo Finance or offline with a local CSV that follows Yahoo's export format.

## Requirements
- Python 3.9 or newer.
- Optional: internet access for live downloads. The script also works offline when you supply a local dataset.

## Quick Start
Run the CLI with a ticker symbol to fetch the last year of daily prices and print a summary:

```bash
python scripts/analyze_stock.py AAPL
```

View the built-in help for all options:

```bash
python scripts/analyze_stock.py --help
```

## Working Offline
A sample Yahoo Finance export (`data/sample_aapl.csv`) is bundled for offline experiments. Combine it with custom moving-average windows:

```bash
python scripts/analyze_stock.py AAPL --data-file data/sample_aapl.csv --moving-averages 5 10 20
```

## Exporting Results
Include the `--export` flag to write the enriched dataset (with moving averages and daily returns) to a CSV file:

```bash
python scripts/analyze_stock.py TSLA --start 2023-01-01 --export data/tsla_analysis.csv
```

## Features
- Downloads daily OHLC data from Yahoo Finance with configurable start and end dates.
- Computes daily returns, multiple moving averages, and summary statistics such as cumulative return and annualised volatility.
- Supports offline analysis by ingesting local CSV files.
- Exports augmented datasets for further analysis in spreadsheets or notebooks.

## Suggested Local Checks
To verify the CLI in your environment:
1. Display the help text: `python scripts/analyze_stock.py --help`
2. Run against the sample offline dataset: `python scripts/analyze_stock.py AAPL --data-file data/sample_aapl.csv`
3. (Optional) Fetch live data if you have network access: `python scripts/analyze_stock.py MSFT --start 2023-01-01`

These commands should complete without errors, printing a summary to the console and generating any requested export files.
