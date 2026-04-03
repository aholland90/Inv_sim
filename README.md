# Min/Max Inventory Simulator

A Streamlit app that simulates store-level inventory over time using consumption data, Min/Max replenishment settings, and order schedules. Includes two recommendation engines — OG (z-score) and Percentile (4-day rolling demand) — to help right-size Min/Max levels by store and SKU.

## Features

- **Simulation** — day-by-day inventory simulation tracking on-hand inventory, orders, receipts, and missed repairs per store/SKU
- **Min/Max Recommendations** — two calculation modes (OG and Percentile) with configurable settings
- **Order Schedule** — weekly order averages by store and day of week
- **Summary & Charts** — fill rate, order patterns, velocity analysis, and pivot tables by month, store, and day of week
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

## Recommendation Modes

### OG (Z-Score)
Uses a statistical safety stock formula driven by a target fill rate slider (default 95%).

- **RecMin** = `ADD × lead time + safety stock`
- **Safety Stock** = `z × σ × √(lead time)` where z is derived from the target fill rate
- **RecMax** = `RecMin + ADD × avg order cycle days`
- **ADD** = total consumed ÷ 90 days

### Percentile (4-Day Rolling Demand)
Builds a forward-looking 4-day rolling demand distribution for every store/part combination and uses percentiles to set Min/Max levels.

**How it works:**
1. Expands the dataset so every store/part has a row for every date in `All_Dates` (missing days filled with 0)
2. For each date, calculates the sum of demand on that day + the next 3 consecutive days
3. Collects all 4-day rolling demand values into a distribution per store/part
4. Applies user-selected percentiles to set RecMin and RecMax

**Sliders:**
- **Min Percentile** (default 50th) — reorder point; half the time 4-day demand is at or below this level
- **Max Percentile** (default 95th) — order-up-to level; covers 95% of observed 4-day demand scenarios

**Note:** At the end of the dataset where fewer than 3 future days remain, incomplete windows are summed from whatever days are available rather than being excluded.

### Fill Rate Cap (both modes)
If a SKU's current fill rate already meets or exceeds the target, RecMin and RecMax will never be recommended higher than current levels — the formula can only suggest holding steady or reducing.
