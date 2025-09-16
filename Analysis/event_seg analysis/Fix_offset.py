import os
import pandas as pd

# Paths
base_dir = r"C:\Users\Smallwood Lab\friends-event-segmentation"
event_seg_dir = os.path.join(base_dir, "Tasks", "event_seg")
log_dir = os.path.join(base_dir, "Tasks", "log_file")
output_dir = os.path.join(base_dir, "Tasks", "event_seg", "Corrected_with_details")
os.makedirs(output_dir, exist_ok=True)

def get_movie_start(log_file):
    """Extract Task Start and Movie Start times by reading log file line by line."""
    task_start = None
    movie_start = None

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Movie Task Start"):
                    try:
                        task_start = float(line.split(",")[1])
                    except:
                        pass
                elif line.startswith("Movie Start"):
                    try:
                        movie_start = float(line.split(",")[1])
                    except:
                        pass
    except Exception as e:
        print(f"Error reading {log_file}: {e}")

    return task_start, movie_start

# Process boundary files
for fname in os.listdir(event_seg_dir):
    if not fname.endswith("_boundaries.csv"):
        continue

    boundary_file = os.path.join(event_seg_dir, fname)
    parts = fname.split("_")
    participant_id = parts[0]
    seed = parts[2]

    log_file = os.path.join(log_dir, f"output_log_{participant_id}_{seed}_full.csv")
    if not os.path.exists(log_file):
        print(f"Skipping {fname}, log file not found: {log_file}")
        continue

    task_start, movie_start = get_movie_start(log_file)
    if movie_start is None or task_start is None:
        print(f"Skipping {fname}, no Movie Start or Task Start found in log.")
        continue

    df = pd.read_csv(boundary_file)
    if "BoundaryTime(s)" not in df.columns:
        print(f"Skipping {fname}, no BoundaryTime(s) column.")
        continue

    corrected_rows = []
    for _, row in df.iterrows():
        original = row["BoundaryTime(s)"]
        corrected = original + (movie_start - task_start)
        corrected_rows.append({
        "ParticipantID": row["ParticipantID"],
        "VideoName": row["VideoName"],
        "BoundaryTime(s)": corrected
    })

    df_corrected = pd.DataFrame(corrected_rows)
    df_corrected = df_corrected.sort_values(by="BoundaryTime(s)")

    out_file = os.path.join(output_dir, fname.replace("_boundaries.csv", "_boundaries.csv"))
    df_corrected.to_csv(out_file, index=False)

    print(f"Corrected file saved to: {out_file}")
