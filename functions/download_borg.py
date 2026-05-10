# download_borg.py
# Downloads Google Clusterdata 2019 from public GCS bucket using only
# google-cloud-storage — no BigQuery quota used at all.
#
# Install: pip install google-cloud-storage tqdm
# The bucket https://console.cloud.google.com/storage/browser/clusterdata_2019_e

import os
from google.cloud import storage
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
CELL        = "e"                  # cluster cell: a, b, c, d, e, f, g, h
# TABLE       = "instance_usage"    
# TABLE       = "instance_events"     
# TABLE       = "machine_events"     
# TABLE       = "collection_events"     
TABLE       = "machine_attributes"     

OUTPUT_DIR  = f"./borg_data/{CELL}/{TABLE}"
BUCKET_NAME = f"clusterdata_2019_{CELL}"
PREFIX      = f"{TABLE}-"          # files: e.g. instance_usage-0000.json.gz
SUFFIX      = ".parquet.gz"           # ".json.gz" or ".parquet"
MAX_FILES   = 3                    # set to e.g. 10 to download only 10 files
                                   # set to None to download everything
# ────────────────────────────────────────────────────────────────────────────

def download_borg(cell, table, output_dir, max_files=None):
    os.makedirs(output_dir, exist_ok=True)

    # Anonymous client — no credentials needed for public buckets
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET_NAME)

    blobs = [b for b in client.list_blobs(bucket, prefix=PREFIX) if b.name.endswith(SUFFIX)]
    if max_files:
        blobs = blobs[:max_files]

    print(f"Found {len(blobs)} files in gs://{BUCKET_NAME}/{PREFIX}")
    print(f"Saving to: {output_dir}\n")

    for blob in tqdm(blobs, unit="file"):
        filename = os.path.basename(blob.name)
        dest_path = os.path.join(output_dir, filename)

        # Skip already-downloaded files (safe to re-run)
        if os.path.exists(dest_path):
            tqdm.write(f"  skip (exists): {filename}")
            continue

        blob.download_to_filename(dest_path)
        tqdm.write(f"  downloaded: {filename}  ({blob.size / 1024**2:.1f} MB)")

    print(f"\nDone. Files saved to: {output_dir}")


if __name__ == "__main__":
    download_borg(CELL, TABLE, OUTPUT_DIR, max_files=MAX_FILES)