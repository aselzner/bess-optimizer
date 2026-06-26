# BESS Dispatch Optimizer

A portfolio project demonstrating energy storage dispatch optimization using real NYISO market data.

## Stack
- **Data**: NYISO Day-Ahead Market LMPs via public bulk API → SQLite
- **Optimization**: Linear programming with PuLP (CBC solver)
- **Visualization**: Matplotlib P&L and dispatch dashboards

## Modules (built incrementally)
| File | Status | Description |
|------|--------|-------------|
| `data_fetch.py` | ✅ Done | Fetch & cache NYISO DAM LMPs |
| `optimizer.py` | 🔜 Next | LP dispatch optimizer |
| `reporting.py` | 🔜 | P&L charts and summary stats |

## Quickstart

```bash
# 1. Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/bess-optimizer.git
cd bess-optimizer

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate mock data and run the test notebook
jupyter notebook 01_test_data_fetch.ipynb

# 5. (Optional) Fetch real NYISO data
python data_fetch.py --start 2025-01-01 --end 2025-01-31 --node CAPITL
```

## Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `node` | `CAPITL` | NYISO pricing zone |
| `battery_mw` | 10 | Power capacity (MW) |
| `battery_mwh` | 40 | Energy capacity (MWh), implies 4-hour duration |
| `rte` | 0.85 | Round-trip efficiency |
| `soc_min` | 0.10 | Minimum state of charge (fraction) |
| `soc_max` | 0.90 | Maximum state of charge (fraction) |

## NYISO Zones
`CAPITL` · `CENTRL` · `DUNWOD` · `GENESE` · `HUD VL` · `LONGIL` · `MHK VL` · `MILLWD` · `N.Y.C.` · `NORTH` · `WEST`
