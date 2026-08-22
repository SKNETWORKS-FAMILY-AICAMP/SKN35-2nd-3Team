import pandas as pd

bus_path = r"C:\sk-encoa\SKN35-2nd-3team\data\raw\bus_stops_nationwide.csv"
subway_path = r"C:\sk-encoa\SKN35-2nd-3team\data\raw\subway_stations_nationwide.xlsx"
out_path = r"C:\sk-encoa\SKN35-2nd-3team\scripts\inspect_transit_result.txt"

with open(out_path, "w", encoding="utf-8") as f:
    for enc in ["utf-8", "cp949", "utf-8-sig"]:
        try:
            bus = pd.read_csv(bus_path, encoding=enc, nrows=5)
            f.write(f"[bus_stops] encoding={enc}\n")
            f.write(f"columns ({len(bus.columns)}): {bus.columns.tolist()}\n")
            f.write(bus.head(3).to_string())
            f.write("\n\n")
            break
        except Exception as e:
            f.write(f"[bus_stops] FAIL {enc}: {type(e).__name__} {str(e)[:150]}\n")

    subway = pd.read_excel(subway_path, nrows=5)
    f.write(f"[subway_stations]\n")
    f.write(f"columns ({len(subway.columns)}): {subway.columns.tolist()}\n")
    f.write(subway.head(3).to_string())
    f.write("\n")

print("done")
