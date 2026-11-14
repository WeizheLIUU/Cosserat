# PLS extension changes

Paths below are relative to the Cosserat plugin root. For setup and example entry points, see the README. For equations, see [PLS formulation](PLS_FORMULATION.md).

## Added components

| Files | Purpose |
| --- | --- |
| `src/Cosserat/mapping/PiecewiseLinearStrainMapping.h/.cpp` | Vec3/Vec6 nodal strain interpolation, velocity propagation, force and constraint transposes, and sparse Jacobians |
| `src/Cosserat/mapping/PLSKinematics.h` | Shared PLS geometry, derivative and visualization calculations |
| `src/Cosserat/forcefield/PLSInertiaForceField.h/.cpp` | Velocity-dependent inertia forces using existing mass and mapping data |

## Modified existing files

| Files | Change |
| --- | --- |
| `src/Cosserat/mapping/Strain2FramesCosseratMapping.h/.inl` | Add the `plsKinematics` branch for PLS geometry and derivatives; handle output frames spanning multiple integration intervals; the option defaults to false, preserving the original calculation path |
| `src/Cosserat/initCosserat.cpp` | Register the two new components |
| `CMakeLists.txt` | Include the new sources and PLS test target |
| `python/cosserat/utils.py` | Optional explicit rigid-frame links and sphere radius in the existing collision helpers; existing default calls retained |

The Jacobian is updated from the current geometry. The implicit solve omits complete geometric stiffness and disables the original partial contribution in the PLS branch. See the formulation document for this approximation.

SOFA core code is unchanged. Material, point-mass, rigid-mapping, collision and constraint components are reused.

## Examples and helpers

Files are in `examples/python3/pls/`.

| File | Purpose |
| --- | --- |
| `pls_interactive.py` | Fixed-base rod under gravity with floor contact |
| `pls_tip_extension.py` | Constant tip force, without gravity or collision |
| `pls_gravity_oscillation.py` | Fixed-base gravity oscillation without collision |
| `pls_free_fall.py` | Free-base rod falling onto a floor |
| `pls_tapered_cantilever.py` | Tapered cantilever with two-point Gauss quadrature |
| `pls_visual.py` | Material-section visualization and output-frame visibility following the native root display flag |

All five examples use the original getting-started header, base and frame
helpers with `CosseratGeometry`. They build native
solvers explicitly. The two contact examples call the existing contact and
collision helpers. The tapered example passes its nonuniform coordinates to
the frame helper and supplies its local material and mass quadrature.
The PLS mechanics helper is `python/cosserat/pls.py`. Internal quadrature
states carry the native `NoPicking` tag. Segmented visualization follows the
output frames. Existing PCS prefabs are unchanged.

The PLS geometry branch locates each output frame across all integration
intervals using the existing metadata helper. Force transposition traverses
all skipped intervals. The default branch retains its original one-interval
behavior. Output resolution therefore does not change the PLS mechanics.

## Tests

- `tests/unit/PiecewiseLinearStrainMapping_test.cpp`: interpolation contracts and material integration.
- `tests/run_pls_integration.py`: geometry, mass, loads, contact and constraint integration, including sparse output frames and virtual-work consistency.
- `tests/run_pls_inertia.py`: velocity-dependent inertia.
