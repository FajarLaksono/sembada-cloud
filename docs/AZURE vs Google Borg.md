Now we have exact numbers for both. Here's the definitive side-by-side:

gsutil du -sh "gs://clusterdata_2019_e/*.parquet.gz"
Google recommends using Gcloud storage CLI (https://docs.cloud.google.com/storage/docs/discover-object-storage-gcloud) instead of gsutil. Please refer to migration guide (https://docs.cloud.google.com/storage/docs/gsutil-transition-to-gcloud) for assistance.
356.1 GiB    gs://clusterdata_2019_e/*.parquet.gz

Complete Dataset Comparison: Google Borg Cell e vs Azure V2

| Metric | Google Borg Cell e | Azure V2 |
|--------|--------------------|----------|
| Compressed size | 356.1 GiB (~383 GB) | 156 GB |
| Uncompressed size | ~1 TB (est.) | 235 GB |
| Format | Parquet.gzCSV.gz |
| Number of files | Thousands of shards | 198 files |
| Duration 31 days (May 2019) | 30 days | 
| Total VMs/instances | Millions (exact unknown w/o query) | 2,695,548 VMs |
| CPU timeseries readings | ~billions (5-min intervals) | 1,942,780,023 readings | 
| Total VM/instance hours | Unknown w/o query | 104,371,713 VM hours
| Virtual core hours | Unknown w/o query | >380,000,000 | 
| CPU granularity | 5-min + full percentile histogram | 5-min (min/avg/max only) | 
| Memory data | ✅ Exact normalized values | ⚠️ Bucket ranges only (4/8/32 GB) |
| Cost to download | $0 | $0 | 
| Cross-table joinability| ⚠️ Broken on partial download | ✅ vmtable is self-contained | 
| Practical local ML | ⚠️ Needs BigQuery extraction first | ✅ Direct, works on any laptop | 
| Subject matter | Container/task scheduling (Borg) | VM lifecycle (closer to thesis) |


Size Ratio
Borg cell e:  356.1 GiB  ████████████████████████░░░░░░░░
Azure V2:     145.0 GiB  ██████████████░░░░░░░░░░░░░░░░░░
                         ~2.46× larger

For Your Thesis (Sembada Cloud)
Azure V2 wins on every practical dimension for a master's thesis:

2.46× smaller — fits comfortably on a standard laptop
Self-contained — vmtable.csv.gz (417 MB) alone is enough for over-provisioning/waste analysis
VM-native — directly models the IaaS layer your thesis targets, unlike Borg's container scheduler
Richer aggregate stats — avgcpu, maxcpu, p95maxcpu per VM are ready-made waste features
No join complexity — Azure's hash sharding problem only affects the 195 CPU timeseries files, not the core vmtable

The only area where Borg wins is CPU distribution richness — its per-5-min percentile histograms are far more detailed than Azure's min/avg/max. If your ML model needs full CPU distributions, Borg via BigQuery extraction is worth the extra step. Otherwise, Azure V2 is the cleaner choice.