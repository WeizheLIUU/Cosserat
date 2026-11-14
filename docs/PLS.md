# Using the PLS extension

`PiecewiseLinearStrainMapping` adds continuous nodal strains to Cosserat. The examples use Vec6 for extension, shear, bending and torsion, with `Strain2FramesCosseratMapping`, `EulerImplicitIntegrationScheme`, `SparseLDLSolver` and `GenericConstraintCorrection`.

## Running an example

From a SOFA workspace configured with Cosserat and SofaPython3:

```bash
./build/bin/runSofa external/Cosserat/examples/python3/pls/pls_interactive.py
```

The scene uses the native SOFA GUI. Select Animate to start. Replace the filename to run another example listed in [PLS changes](PLS_CHANGES.md).

Uniform-rod examples default to 10 PLS sections, three geometry intervals per section and a 0.002 s time step. Parameters are at the top of each scene.

## Scene construction

All five PLS examples follow the upstream `getting_started/using_lie_algebra`
scene layout. They call
`add_mini_header`, `_add_rigid_base` and `_add_cosserat_frame_v2`, build the
native solvers explicitly, and pass a `CosseratGeometry` to the frame helper.
The fixed-base examples apply their existing fixed constraints, and the
contact examples use the native `contactHeader` and `addPointsCollision`.
PLS-specific code supplies nodal strains, midpoint geometry samples and
quadrature mass. The tapered example supplies its nonuniform coordinates,
local radii and viscous damping. All scenes use the segmented visualization.
Output density is independent of the integration discretization.

Internal mass-quadrature states carry SOFA's standard `NoPicking` tag.
Hiding a MechanicalObject does not exclude it from native ray picking.
No custom mouse operation or matrix-system adapter is installed.

The PLS examples show the complete construction directly. The shared mechanics
helper is `python/cosserat/pls.py`; the existing PCS prefabs are unchanged.

## Component data

- `nodeAbscissa`: finite, strictly increasing material coordinates, one per nodal strain.
- `sampleAbscissa`: sampling coordinates within the nodal domain.
- `applyRestPosition`: interpolate rest strains; enabled by default.

Repeated, unordered or empty samples are supported. Extrapolation is rejected. Invalid inputs mark the component invalid; corrected inputs permit recovery.

## Mass and inertia

Mass uses two axial Gauss points per geometry interval and four points per circular section. Existing mass components and mappings assemble the configuration-dependent generalized mass. `PLSInertiaForceField` supplies the velocity-dependent inertia bias on the same points. This bias is evaluated explicitly; the solve does not include its complete implicit tangent.

## Display

`pls_visual.py` draws one colored tube per PLS section using `OglModel` and `RigidMapping`. Set `SEGMENT_COLORS`, `VISUAL_RING_POINTS` and `SHOW_SEGMENT_BOUNDARIES` at the top of the scene. Visual mappings do not transfer forces, constraints or mass.

Output-frame axes are hidden by default. `pls_visual.py` synchronizes their
`showObject` data with the root **Show Behavior Models** flag during drawing,
including while animation is paused. The axis scale is one rod radius
(`RADIUS_BASE` for the taper); mass-quadrature frames remain hidden. This visual
synchronization is needed because native `MechanicalObject.showObject` draws
independently of that display flag in the supported SOFA runtime.

See [formulation](PLS_FORMULATION.md) for equations and the geometric-stiffness approximation, and [changes](PLS_CHANGES.md) for the file-level implementation summary.
