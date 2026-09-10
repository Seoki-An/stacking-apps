# Tabletop runtime data

This directory contains local or generated tabletop data. Its contents are
ignored by Git except for this README. Curated object models belong in
`../assets/objects/`, not here.

The expected layout is:

```text
data/
├── captures/                 # Raw camera PCDs and synchronized sensor records
├── reconstruction/
│   └── <object-id>/
│       └── <session-id>/     # Manifest, merged PCD, mesh, and DSF output
└── evaluation/               # Recorded inputs and benchmark results
```

Reconstruction sessions are self-contained below their session directory. The
`session.json` manifest records accepted views, pose-graph transforms, merge
statistics, inertial metadata, and paths to generated outputs.

Large datasets may be stored outside the repository by passing an explicit
session directory to the reconstruction CLI. After review, an approved object
catalog entry should be copied into `../assets/objects/` through a controlled
promotion workflow.
