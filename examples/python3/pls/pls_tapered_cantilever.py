"""Vec6 PLS tapered rod with the reference parameters (SI units).

From the workspace root:
    ./build/bin/runSofa external/Cosserat/examples/python3/pls/pls_tapered_cantilever.py

Three PLS elements, 20 midpoint geometry intervals, two-point Gauss integration
for material and mass. Gravity drives the fixed-base rod without contact.
The zero-load reference boundary fixes the last strain node, not the tip position.
"""
import math
from pathlib import Path
import sys

PLUGIN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PLUGIN / "python"))
sys.path.insert(0, str(PLUGIN / "tutorials/getting_started/using_lie_algebra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosserat.geometry import CosseratGeometry
from cosserat.params import BeamGeometryParameters
from pls_visual import add_segmented_visual


# Reference geometry and material. DISCS includes both ends of each element.
LENGTHS = [0.09, 0.07, 0.04]
DISCS = [10, 8, 5]
RADIUS_BASE = 0.01
RADIUS_TIP = 0.005
YOUNG_MODULUS = 110000.0
POISSON_RATIO = 0.45
VISCOSITY = 800.0
DENSITY = 2000.0
TIME_STEP = 0.001
GRAVITY = [0.0, 0.0, -9.81]
RAYLEIGH_MASS = 0.0
RAYLEIGH_STIFFNESS = 0.0
ENABLE_INERTIA_BIAS = True
CAMERA_POSITION = [0.1, -0.42, 0.09]
CAMERA_TARGET = [0.08, 0.0, -0.06]

SEGMENT_COLORS = [[0.12, 0.42, 0.72, 1.0], [0.12, 0.72, 0.62, 1.0]]
VISUAL_RING_POINTS = 16
SHOW_SEGMENT_BOUNDARIES = True


def radius_at(s):
    """Linear taper in the reference arc-length coordinate."""
    return RADIUS_BASE + (RADIUS_TIP - RADIUS_BASE) * s / sum(LENGTHS)


def discretization():
    """Return strain nodes, geometry boundaries and reference output positions."""
    nodes = [0.0]
    for length in LENGTHS:
        nodes.append(nodes[-1] + length)
    # Keep repeated element interfaces in the reference output ordering only.
    outputs = [
        nodes[e] + LENGTHS[e] * j / (DISCS[e] - 1)
        for e in range(len(LENGTHS)) for j in range(DISCS[e])
    ]
    sections = [nodes[0]] + [
        nodes[e] + LENGTHS[e] * j / (DISCS[e] - 1)
        for e in range(len(LENGTHS)) for j in range(1, DISCS[e])
    ]
    return nodes, sections, outputs


def gauss2_quadrature(boundaries):
    """Two physical (position, weight) pairs per interval."""
    gauss_nodes = [0.5 - 1.0 / (2 * math.sqrt(3.0)),
                   0.5 + 1.0 / (2 * math.sqrt(3.0))]
    return [
        (left + (right - left) * xi, (right - left) / 2)
        for left, right in zip(boundaries[:-1], boundaries[1:])
        for xi in gauss_nodes
    ]


def createScene(root):
    from introduction_and_setup import add_mini_header, _add_rigid_base, _add_cosserat_frame_v2

    add_mini_header(root)
    root.addObject("RequiredPlugin", name="dynamicsPlugins", pluginName=[
        "Sofa.Component.AnimationLoop", "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.IntegrationScheme.Backward", "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSystem", "Sofa.Component.Mapping.Linear",
        "Sofa.Component.MechanicalLoad", "Sofa.GUI.Component",
    ])
    root.removeObject(root.getObject("DefaultAnimationLoop"))
    root.addObject("FreeMotionAnimationLoop")
    root.addObject("RequiredPlugin", name="materialPlugins", pluginName=[
        "Sofa.Component.Topology.Container.Constant", "Sofa.Component.SolidMechanics.Spring",
    ])
    root.addObject("BlockGaussSeidelConstraintSolver", name="ConstraintSolver",
                   tolerance=1e-9, maxIterations=200)
    root.dt = TIME_STEP
    root.gravity = GRAVITY
    root.VisualStyle.displayFlags = "showVisualModels hideBehaviorModels hideMechanicalMappings hideCollisionModels"
    root.addObject("InteractiveCamera", name="camera", position=CAMERA_POSITION,
                   lookAt=CAMERA_TARGET, zNear=0.001, zFar=5.0)

    solver = root.addChild("rod")
    solver.addObject("EulerImplicitIntegrationScheme", firstOrder=False,
                     rayleighMass=RAYLEIGH_MASS, rayleighStiffness=RAYLEIGH_STIFFNESS)
    solver.addObject("SparseLDLSolver", name="solver", template="CompressedRowSparseMatrixd")
    solver.addObject("GenericConstraintCorrection", linearSolver="@solver")
    base = _add_rigid_base(solver)
    base.cosserat_base_mo.showObject = False
    base.removeObject(base.spring)
    base_state = base.cosserat_base_mo
    base.addObject("FixedProjectiveConstraint", indices=[0])

    # Four Vec6 nodes. The reference zero-load boundary fixes the last strain.
    nodes, sections, frame_abscissa = discretization()
    strain = solver.addChild("cosserat_states")
    nodal_state = strain.addObject("MechanicalObject", template="Vec6d", name="cosserat_state",
                                   position=[[0.0] * 6 for _ in nodes])
    strain.addObject("FixedProjectiveConstraint", name="referenceTipStrain", indices=[len(nodes) - 1])

    # Two material samples per PLS element. Separate native force fields allow
    # each sample to use its local radius; weights carry the integration length.
    material = strain.addChild("material")
    for k, (s, weight) in enumerate(gauss2_quadrature(nodes)):
        radius = radius_at(s)
        area = math.pi * radius * radius
        inertia = math.pi * radius**4 / 4
        group = material.addChild(f"quadrature{k}")
        point_state = group.addObject("MechanicalObject", template="Vec6d", name="state")
        group.addObject("PiecewiseLinearStrainMapping", template="Vec6d",
                        input=nodal_state.getLinkPath(), output=point_state.getLinkPath(),
                        nodeAbscissa=nodes, sampleAbscissa=[s])
        group.addObject("BeamHookeLawForceField", name="elasticity", template="Vec6d",
                        crossSectionShape="circular", length=[weight], radius=radius,
                        youngModulus=YOUNG_MODULUS, poissonRatio=POISSON_RATIO)
        coefficients = [VISCOSITY * weight * value for value in
                        [2 * inertia, 3 * inertia, 3 * inertia, 3 * area, area, area]]
        group.addObject("DiagonalVelocityDampingForceField", template="Vec6d", name="viscosity",
                        dampingCoefficient=[coefficients])

    # Geometry uses 20 dependent midpoint samples, not additional strain DOFs.
    integration = strain.addChild("integration")
    integration_state = integration.addObject("MechanicalObject", template="Vec6d", name="cosserat_state")
    midpoints = [(a + b) / 2 for a, b in zip(sections[:-1], sections[1:])]
    integration.addObject("PiecewiseLinearStrainMapping", template="Vec6d",
                          input=nodal_state.getLinkPath(), output=integration_state.getLinkPath(),
                          nodeAbscissa=nodes, sampleAbscissa=midpoints)
    geometry = CosseratGeometry(BeamGeometryParameters(
        beam_length=nodes[-1], nb_section=len(sections) - 1, nb_frames=len(frame_abscissa) - 1))
    geometry.curv_abs_sections = sections
    geometry.section_lengths = [b - a for a, b in zip(sections[:-1], sections[1:])]
    geometry.curv_abs_frames = frame_abscissa
    geometry.update_frames([[s, 0, 0, 0, 0, 0, 1] for s in frame_abscissa])
    frames = _add_cosserat_frame_v2(base, integration, geometry)
    frames.cosseratMapping.plsKinematics = True
    frames.cosseratMapping.radius = RADIUS_BASE
    frames.FramesMO.showObject = False
    frames.FramesMO.showIndices = False
    frames.FramesMO.showObjectScale = 1 * RADIUS_BASE
    frames.removeObject(frames.getObject("UniformMass"))

    # Two mass samples per geometry interval, four points per circular section.
    # Cartesian DiagonalMass is mapped to the coupled strain-space mass matrix.
    mass_samples = gauss2_quadrature(sections)
    mass_abscissas = [0.0] + [s for s, weight in mass_samples]
    mass_frames = base.addChild("massFrames")
    integration.addChild(mass_frames)
    mass_state = mass_frames.addObject("MechanicalObject", template="Rigid3d", name="state",
        position=[[s, 0, 0, 0, 0, 0, 1] for s in mass_abscissas], tags="NoPicking")
    mass_frames.addObject("Strain2FramesCosseratMapping", name="geometry", plsKinematics=True,
        input1=integration_state.getLinkPath(), input2=base_state.getLinkPath(),
        output=mass_state.getLinkPath(), curv_abs_input=sections,
        curv_abs_output=mass_abscissas, radius=RADIUS_BASE, nonColored=True)
    local_points = []
    masses = []
    indices = []
    for k, (s, weight) in enumerate(mass_samples):
        radius = radius_at(s)
        offset = radius / math.sqrt(2.0)
        local_points.extend([[0, offset, 0], [0, -offset, 0],
                             [0, 0, offset], [0, 0, -offset]])
        masses.extend([DENSITY * math.pi * radius * radius * weight / 4] * 4)
        indices.extend([k + 1] * 4)

    group = mass_frames.addChild("quadrature")
    point_state = group.addObject("MechanicalObject", template="Vec3d", name="state",
                                  position=local_points, tags="NoPicking")
    # Edges meet the native topology contract; explicit vertexMass sets weights.
    group.addObject("MeshTopology", name="topology", position=local_points,
                    edges=[[i, i + 1] for i in range(len(local_points) - 1)])
    group.addObject("DiagonalMass", name="mass", template="Vec3d,Vec3d", vertexMass=masses,
                    topology="@topology", geometryState="@state")
    group.addObject("RigidMapping", name="sectionPoints", input=mass_state.getLinkPath(),
                    output=point_state.getLinkPath(), rigidIndexPerPoint=indices,
                    globalToLocalCoords=False)
    if ENABLE_INERTIA_BIAS:
        group.addObject("PLSInertiaForceField", name="inertiaBias",
            cosseratMapping=mass_frames.geometry.getLinkPath(),
            rigidMapping=group.sectionPoints.getLinkPath(), mass=group.mass.getLinkPath())

    # Material-element colors follow the existing reference frames and taper.
    add_segmented_visual(frames, frame_abscissa, nodes, radius_at,
                         colors=SEGMENT_COLORS, ring_points=VISUAL_RING_POINTS,
                         show_boundaries=SHOW_SEGMENT_BOUNDARIES, frame_state=frames.FramesMO)
    return root
