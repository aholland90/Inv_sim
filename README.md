# Min/Max Inventory Simulator

A Streamlit app that simulates store-level inventory over time using consumption data, Min/Max replenishment settings, and order schedules.

## Features

- **Simulation** — models daily inventory, orders, receipts, and missed repairs per store/SKU
- **Min/Max Recommendations** — z-score based safety stock recommendations at a configurable target fill rate
- **Order Schedule** — weekly order averages by store and day of week
- **Summary & Charts** — fill rate, order patterns, velocity analysis, and pivot tables by month/store/day of week
- **Low-velocity SKU flagging** — configurable threshold to identify slow-moving parts

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the app:
   ```bash
   streamlit run inventory_sim.py
   ```

## Excel Upload Format

Upload a single `.xlsx` file with the following sheets:

| Sheet | Required Columns |
|---|---|
| `Consumed` | `ConDate`, `StoreNum`, `PartNum`, `Consumption` |
| `Min_Max_Store_SKU` | `StoreNum`, `PartNum`, `Min`, `Max` |
| `Store_Order_Schedule` | `StoreNum`, order day columns |
| `All_Dates` | `ConDate` (date spine for simulation) |
| `Store_List` *(optional)* | `StoreNum`, `StoreDesc` |

## Simulation Logic

- Inventory is consumed first, then an order check runs
- Orders are placed when `InventoryOH ≤ Min` on an allowed order day
- Order quantity = `Max − current inventory`
- Orders placed today are received after the configured lead time
- `MissedRepairs` = demand that could not be fulfilled due to zero inventory
- Day of week: `1 = Sunday … 7 = Saturday`
