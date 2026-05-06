"""
Google ClusterData 2019 - Fetch Functions

Fetches data from Google Borg cluster traces (2019) via BigQuery.
Dataset: https://github.com/google/cluster-data/blob/master/ClusterData2019.md

Usage:
    from functions.fetch_cluster_data import (
        get_bigquery_client,
        fetch_machine_capacity_sample,
        fetch_collection_events_sample,
        fetch_instance_usage_sample,
        fetch_cell_summary,
    )

    client = get_bigquery_client(project_id="your-gcp-project")
    result = fetch_machine_capacity_sample(client, cell="a", limit=10)
    print(result.dataframe)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

# Load .env file from project root (auto-detects up the directory tree)
load_dotenv()

# BigQuery public dataset location
BQ_PROJECT = "google.com:google-cluster-data"
BQ_DATASET_PREFIX = "clusterdata_2019_"
VALID_CELLS = list("abcdefgh")


@dataclass
class FetchResult:
    """Container for fetch operation results."""

    dataframe: pd.DataFrame
    query: str
    row_count: int
    cell: str
    table: str


def get_bigquery_client(project_id: Optional[str] = None) -> bigquery.Client:
    """
    Create and return a BigQuery client.

    Args:
        project_id: GCP project ID. Falls back to GCP_PROJECT_ID env var.

    Returns:
        Authenticated BigQuery client.
    """
    pid = project_id or os.environ.get("GCP_PROJECT_ID")
    if not pid:
        raise ValueError(
            "project_id not provided and GCP_PROJECT_ID env var not set. "
            "Set it with: set GCP_PROJECT_ID=your-project-id"
        )
    return bigquery.Client(project=pid)


def _table_name(cell: str, table: str) -> str:
    """Build fully qualified BigQuery table name."""
    if cell not in VALID_CELLS:
        raise ValueError(f"Cell must be one of {VALID_CELLS}, got '{cell}'")
    return f"`{BQ_PROJECT}.{BQ_DATASET_PREFIX}{cell}.{table}`"


def fetch_machine_capacity_sample(
    client: bigquery.Client, cell: str = "a", limit: int = 10
) -> FetchResult:
    """
    Fetch a sample of machine capacity data.

    Args:
        client: Authenticated BigQuery client.
        cell: Borg cell identifier (a-h).
        limit: Maximum rows to return.

    Returns:
        FetchResult with dataframe and metadata.
    """
    table = _table_name(cell, "machine_events")
    query = f"""
    SELECT
        machine_id,
        capacity.cpus AS cpu_capacity,
        capacity.memory AS memory_capacity,
        platform_id
    FROM {table}
    LIMIT {limit}
    """
    df = client.query(query).to_dataframe()
    return FetchResult(
        dataframe=df,
        query=query,
        row_count=len(df),
        cell=cell,
        table="machine_events",
    )


def fetch_collection_events_sample(
    client: bigquery.Client, cell: str = "a", limit: int = 10
) -> FetchResult:
    """
    Fetch a sample of collection (job) events.

    Args:
        client: Authenticated BigQuery client.
        cell: Borg cell identifier (a-h).
        limit: Maximum rows to return.

    Returns:
        FetchResult with dataframe and metadata.
    """
    table = _table_name(cell, "collection_events")
    query = f"""
    SELECT
        collection_id,
        collection_type,
        user,
        priority,
        collection_name,
        resource_request.cpus AS requested_cpus,
        resource_request.memory AS requested_memory
    FROM {table}
    LIMIT {limit}
    """
    df = client.query(query).to_dataframe()
    return FetchResult(
        dataframe=df,
        query=query,
        row_count=len(df),
        cell=cell,
        table="collection_events",
    )


def fetch_instance_usage_sample(
    client: bigquery.Client, cell: str = "a", limit: int = 10
) -> FetchResult:
    """
    Fetch a sample of instance usage records.

    Args:
        client: Authenticated BigQuery client.
        cell: Borg cell identifier (a-h).
        limit: Maximum rows to return.

    Returns:
        FetchResult with dataframe and metadata.
    """
    table = _table_name(cell, "instance_usage")
    query = f"""
    SELECT
        collection_id,
        instance_index,
        machine_id,
        start_time,
        end_time,
        average_usage.cpus AS avg_cpu_usage,
        average_usage.memory AS avg_memory_usage,
        maximum_usage.cpus AS max_cpu_usage,
        maximum_usage.memory AS max_memory_usage
    FROM {table}
    LIMIT {limit}
    """
    df = client.query(query).to_dataframe()
    return FetchResult(
        dataframe=df,
        query=query,
        row_count=len(df),
        cell=cell,
        table="instance_usage",
    )


def fetch_cell_summary(
    client: bigquery.Client, cell: str = "a"
) -> FetchResult:
    """
    Fetch summary statistics for a Borg cell.

    Args:
        client: Authenticated BigQuery client.
        cell: Borg cell identifier (a-h).

    Returns:
        FetchResult with summary dataframe.
    """
    machine_table = _table_name(cell, "machine_events")
    collection_table = _table_name(cell, "collection_events")

    query = f"""
    SELECT
        (SELECT COUNT(DISTINCT machine_id) FROM {machine_table}) AS total_machines,
        (SELECT COUNT(DISTINCT collection_id) FROM {collection_table}) AS total_collections
    """
    df = client.query(query).to_dataframe()
    return FetchResult(
        dataframe=df,
        query=query,
        row_count=len(df),
        cell=cell,
        table="summary",
    )


def main():
    """Run a simple data fetch test."""
    print("=" * 60)
    print("Google ClusterData 2019 - Data Fetch Test")
    print("=" * 60)

    try:
        client = get_bigquery_client()
    except ValueError as e:
        print(f"\nError: {e}")
        return

    # Test 1: Machine capacity sample
    print("\n--- Test 1: Machine Capacity Sample (cell 'a', 5 rows) ---")
    try:
        result = fetch_machine_capacity_sample(client, cell="a", limit=5)
        print(f"Query executed successfully. Rows returned: {result.row_count}")
        print(result.dataframe.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Collection events sample
    print("\n--- Test 2: Collection Events Sample (cell 'a', 5 rows) ---")
    try:
        result = fetch_collection_events_sample(client, cell="a", limit=5)
        print(f"Query executed successfully. Rows returned: {result.row_count}")
        print(result.dataframe.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Cell summary
    print("\n--- Test 3: Cell Summary (cell 'a') ---")
    try:
        result = fetch_cell_summary(client, cell="a")
        print("Query executed successfully.")
        print(result.dataframe.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Data fetch test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
