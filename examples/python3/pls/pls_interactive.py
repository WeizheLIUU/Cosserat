"""Fixed-base rod bending under gravity onto a floor.

From the SOFA workspace root:
    ./build/bin/runSofa external/Cosserat/examples/python3/pls/pls_interactive.py
"""
from pathlib import Path
import sys

PLUGIN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PLUGIN / "python"))
sys.path.insert(0, str(PLUGIN / "tutorials/getting_started/using_lie_algebra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosserat.geometry import CosseratGeometry
from cosserat.params import BeamGeometryParameters
from pls_visual import add_segmented_visual
from cosserat.header import contactHeader
from cosserat.params import ContactParameters
from cosserat.pls import RodParameters, add_strain, add_quadrature_mass
from cosserat.utils import addPointsCollision

STRAIN_COMPONENTS = 6
LENGTH = 0.5
ELEMENT_COUNT = 10
INTEGRATION_SUBSTEPS = 3
RADIUS = 0.01
YOUNG_MODULUS = 2.0e6
POISSON_RATIO = 0.3
TOTAL_MASS = 0.015
TIME_STEP = 0.002
GRAVITY = [0.0, -9.81, 0.0]
TIP_FORCE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # World force (N), then torque (N m).
FIXED_BASE = True
ENABLE_CONTACT = True
FLOOR_HEIGHT = -0.12
FRICTION = 0.2
RAYLEIGH_MASS = 0.2
RAYLEIGH_STIFFNESS = 0.01
ENABLE_INERTIA_BIAS = True

SEGMENT_COLORS = [[0.12, 0.42, 0.72, 1.0], [0.12, 0.72, 0.62, 1.0]]
VISUAL_RING_POINTS = 16
SHOW_SEGMENT_BOUNDARIES = True


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
    root.addObject("BlockGaussSeidelConstraintSolver", name="ConstraintSolver",
                   tolerance=1e-9, maxIterations=200, computeConstraintForces=True)
    if ENABLE_CONTACT:
        root.addObject("RequiredPlugin", name="contactPlugins", pluginName=[
            "Sofa.Component.Collision.Detection.Algorithm",
            "Sofa.Component.Collision.Detection.Intersection",
            "Sofa.Component.Collision.Geometry", "Sofa.Component.Collision.Response.Contact",
            "Sofa.Component.Topology.Container.Constant", "Sofa.Component.Topology.Container.Dynamic",
        ])
        contactHeader(root, ContactParameters(responseParams=f"mu={FRICTION}",
                                              alarmDistance=0.006, contactDistance=0.001))
    root.dt = TIME_STEP
    root.gravity = GRAVITY
    root.VisualStyle.displayFlags = (
        "showVisualModels hideBehaviorModels hideMechanicalMappings hideCollisionModels "
        "hideBoundingCollisionModels hideInteractionForceFields")

    solver = root.addChild("rod")
    solver.addObject("EulerImplicitIntegrationScheme", firstOrder=False,
                     rayleighMass=RAYLEIGH_MASS, rayleighStiffness=RAYLEIGH_STIFFNESS)
    solver.addObject("SparseLDLSolver", name="solver", template="CompressedRowSparseMatrixd")
    solver.addObject("GenericConstraintCorrection", linearSolver="@solver")
    base = _add_rigid_base(solver)
    base.cosserat_base_mo.showObject = False
    base.removeObject(base.spring)
    if FIXED_BASE:
        base.addObject("FixedProjectiveConstraint", indices=[0])

    interval_count = ELEMENT_COUNT * INTEGRATION_SUBSTEPS
    geometry = CosseratGeometry(BeamGeometryParameters(
        beam_length=LENGTH, nb_section=interval_count, nb_frames=interval_count))
    parameters = RodParameters(length=LENGTH, elements=ELEMENT_COUNT,
        substeps=INTEGRATION_SUBSTEPS, radius=RADIUS, young_modulus=YOUNG_MODULUS,
        poisson_ratio=POISSON_RATIO, mass=TOTAL_MASS, dimension=STRAIN_COMPONENTS)
    strain = add_strain(solver, parameters, node_name="cosserat_states",
                        state_name="cosserat_state", integration_state_name="cosserat_state")
    integration = strain.integration
    geometry.curv_abs_sections = strain.sections
    frames = _add_cosserat_frame_v2(base, integration, geometry)
    frames.cosseratMapping.plsKinematics = True
    frames.cosseratMapping.radius = RADIUS
    frames.FramesMO.showObject = False
    frames.FramesMO.showIndices = False
    frames.FramesMO.showObjectScale = 1 * RADIUS
    frames.removeObject(frames.getObject("UniformMass"))
    add_quadrature_mass(base, strain, parameters, base_state=base.cosserat_base_mo,
                        inertia_bias=ENABLE_INERTIA_BIAS)

    frames.addObject("ConstantForceField", name="tipForce", template="Rigid3d",
                     indices=[len(geometry.curv_abs_frames) - 1], forces=[TIP_FORCE])
    add_segmented_visual(frames, geometry.curv_abs_frames, strain.nodes, lambda s: RADIUS,
        colors=SEGMENT_COLORS, ring_points=VISUAL_RING_POINTS,
        show_boundaries=SHOW_SEGMENT_BOUNDARIES, frame_state=frames.FramesMO)

    if ENABLE_CONTACT:
        addPointsCollision(frames, geometry.cable_positions, geometry.get_edge_list(),
                           "CollisionPoints", rigidState=frames.FramesMO, radius=RADIUS)
        floor = root.addChild("floor")
        vertices = [[-0.3, FLOOR_HEIGHT, -0.3], [LENGTH + 0.3, FLOOR_HEIGHT, -0.3],
                    [LENGTH + 0.3, FLOOR_HEIGHT, 0.3], [-0.3, FLOOR_HEIGHT, 0.3]]
        triangles = [[0, 2, 1], [0, 3, 2]]
        floor.addObject("MeshTopology", position=vertices, triangles=triangles)
        floor.addObject("MechanicalObject", template="Vec3d", position=vertices)
        floor.addObject("TriangleCollisionModel", simulated=False, moving=False, group=[3])
        floor.addChild("visual").addObject("OglModel", position=vertices, triangles=triangles,
                                           color=[0.65, 0.7, 0.77, 1.0])
    return root
