# Changelog

## Unreleased — PLS extension

- Add Vec3/Vec6 piecewise linear strain interpolation with force and constraint transposes.
- Add the optional PLS geometry branch to `Strain2FramesCosseratMapping`, including independent output-frame spacing.
- Reuse existing material components with two-point Gauss integration.
- Assemble mass through axial and cross-sectional quadrature points.
- Add `PLSInertiaForceField` for velocity-dependent inertia.
- Add five load/contact examples and PLS-specific construction helpers.
- Use the upstream getting-started helpers for all five PLS examples.
- Add material-section visualization using existing rigid frames.
- Add C++ contracts and Python integration/inertia checks.

See [implementation changes](docs/PLS_CHANGES.md) and [formulation](docs/PLS_FORMULATION.md).
