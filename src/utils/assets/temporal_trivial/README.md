# Temporal Trivial Audio Assets

These WAV files are bundled only for the `dog_car_order_trivial` benchmark family.

- `dog_bark.wav`: procedurally generated bark-like asset created in-repo from shaped noise plus low tonal components
- `car_horn.wav`: procedurally generated horn-like asset created in-repo from steady dual-tone synthesis

Provenance:
- Generated locally for this repository on 2026-03-25 during benchmark implementation
- No third-party recordings are included

License:
- Repository code and generated assets are distributed under the same project license unless stated otherwise elsewhere in the repo

Notes:
- Assets are short, mono, and pre-trimmed
- The benchmark builder normalizes and stitches them but does not transform their semantics
