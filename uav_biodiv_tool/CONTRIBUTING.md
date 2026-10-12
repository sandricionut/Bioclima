# Contributing

Contributions should preserve the package's central rules:

- keep the core sensor- and vendor-agnostic;
- do not introduce hard-coded channel/band names or wavelengths;
- do not infer physical units from variable names;
- preserve physical values and provenance during analysis-ready fusion;
- keep large-data paths bounded in RAM;
- add tests for new functionality;
- keep examples explicit about assumptions and input schemas.

## Development

```bash
python -m pip install -e .
python -m pytest -v
```

Before a release:

```bash
python -m build
```

New point-cloud readers should expose geometry and arbitrary numeric attributes through `PointCloudData` and should support streaming when the format permits it.
