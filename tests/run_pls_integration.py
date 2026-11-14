"""Serial integration checks against the built SOFA/Cosserat runtime.

Run with this workspace's SofaPython3 on PYTHONPATH and SOFA_ROOT=build.
The C++ contracts are covered separately by Cosserat_pls_tests.
"""
import argparse
import contextlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from Sofa import SofaLinearSolver  # Register native sparse access before any scene is created.

PLUGIN = Path(__file__).resolve().parents[1]
SCENE = PLUGIN / "examples/python3/pls/pls_interactive.py"


def check_scene(root):
    pending = [root]
    seen = set()
    errors = []
    while pending:
        node = pending.pop()
        path = node.getPathName()
        if path in seen:
            continue
        seen.add(path)
        for obj in node.objects:
            if "Invalid" in str(obj.componentState.value):
                errors.append(f"{obj.getPathName()}: Invalid")
            messages = obj.getLoggedMessagesAsString()
            if any(line.startswith(("4:", "5:", "Error:", "Fatal:"))
                   for line in messages.splitlines()):
                errors.append(f"{obj.getPathName()}: {messages}")
        pending.extend(node.children)
    assert not errors, "\n".join(errors)


@contextlib.contextmanager
def scene(model="PLS", dimension=3, substeps=8, strains=None, velocities=None,
          before_init=None, **settings):
    spec = importlib.util.spec_from_file_location("pls_validation_scene", SCENE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ELEMENT_COUNT = 5  # Fixed regression fixture, independent of editable demo defaults.
    module.STRAIN_MODEL = model
    module.STRAIN_COMPONENTS = dimension
    module.INTEGRATION_SUBSTEPS = substeps
    for key, value in settings.items():
        setattr(module, key, value)
    root = Sofa.Core.Node("root")
    from cosserat.pls import RodParameters, add_strain, add_frames, add_quadrature_mass
    p = RodParameters(length=module.LENGTH, elements=module.ELEMENT_COUNT,
        substeps=substeps, radius=module.RADIUS, young_modulus=module.YOUNG_MODULUS,
        poisson_ratio=module.POISSON_RATIO, mass=module.TOTAL_MASS, model=model, dimension=dimension)
    root.dt = module.TIME_STEP
    root.gravity = module.GRAVITY
    root.addObject("RequiredPlugin", pluginName=[
        "Cosserat", "Sofa.Component.AnimationLoop", "Sofa.Component.StateContainer",
        "Sofa.Component.IntegrationScheme.Backward", "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mapping.Linear", "Sofa.Component.Mapping.NonLinear", "Sofa.Component.Mass",
        "Sofa.Component.MechanicalLoad", "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.Constraint.Projective", "Sofa.Component.Constraint.Lagrangian.Model",
        "Sofa.Component.Constraint.Lagrangian.Correction", "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Collision.Detection.Algorithm", "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Response.Contact", "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Topology.Container.Constant", "Sofa.Component.Visual"])
    root.addObject("FreeMotionAnimationLoop")
    root.addObject("BlockGaussSeidelConstraintSolver", name="constraints", tolerance=1e-9,
                   maxIterations=200, computeConstraintForces=True)
    rod = root.addChild("rod")
    rod.addObject("EulerImplicitIntegrationScheme", rayleighMass=module.RAYLEIGH_MASS,
                  rayleighStiffness=module.RAYLEIGH_STIFFNESS)
    rod.addObject("SparseLDLSolver", name="linearSolver", template="CompressedRowSparseMatrixd")
    rod.addObject("GenericConstraintCorrection", linearSolver="@linearSolver")
    base = rod.addChild("base")
    base.addObject("MechanicalObject", name="state", template="Rigid3d", position=[[0, 0, 0, 0, 0, 0, 1]])
    if module.FIXED_BASE:
        base.addObject("FixedProjectiveConstraint", indices=[0])
    strain = add_strain(rod, p)
    frames = add_frames(base, strain, p)
    add_quadrature_mass(base, strain, p, inertia_bias=module.ENABLE_INERTIA_BIAS)
    frames.addObject("ConstantForceField", template="Rigid3d", indices=[len(strain.abscissa) - 1],
                     forces=[module.TIP_FORCE])
    if module.ENABLE_CONTACT:
        root.addObject("CollisionPipeline")
        root.addObject("BruteForceBroadPhase")
        root.addObject("BVHNarrowPhase")
        root.addObject("LocalMinDistance", alarmDistance=0.006, contactDistance=0.001)
        root.addObject("RuleBasedContactManager", response="FrictionContactConstraint",
                       responseParams=f"mu={module.FRICTION}")
        points = frames.addChild("surface")
        surface = points.addObject("MechanicalObject", name="state", template="Vec3d",
                                   position=[[0, 0, 0] for _ in strain.abscissa])
        points.addObject("RigidMapping", input=frames.state.getLinkPath(), output=surface.getLinkPath(),
                         rigidIndexPerPoint=list(range(len(strain.abscissa))), globalToLocalCoords=False)
        points.addObject("SphereCollisionModel", radius=p.radius, group=[1], selfCollision=False)
        floor = root.addChild("floor")
        vertices = [[-0.3, module.FLOOR_HEIGHT, -0.3], [p.length + 0.3, module.FLOOR_HEIGHT, -0.3],
                    [p.length + 0.3, module.FLOOR_HEIGHT, 0.3], [-0.3, module.FLOOR_HEIGHT, 0.3]]
        floor.addObject("MeshTopology", position=vertices, triangles=[[0, 2, 1], [0, 3, 2]])
        floor.addObject("MechanicalObject", template="Vec3d", position=vertices)
        floor.addObject("TriangleCollisionModel", simulated=False, moving=False, group=[2])
    state = root.rod.strain.state
    if strains is not None:
        state.position.value = strains.tolist()
        state.rest_position.value = np.zeros_like(strains).tolist()
    if velocities is not None:
        state.velocity.value = velocities.tolist()
    if before_init:
        before_init(root, module)
    try:
        Sofa.Simulation.init(root)
        check_scene(root)
        yield root, module
        check_scene(root)
    finally:
        Sofa.Simulation.unload(root)


def frame_state(root):
    return root.rod.base.frames.state


def animate(root, steps):
    for _ in range(steps):
        Sofa.Simulation.animate(root, float(root.dt.value))
    positions = frame_state(root).position.array()
    assert np.isfinite(positions).all()
    return positions.copy()


def material_energy(dimension):
    rng = np.random.default_rng(4821 + dimension)
    strains = rng.uniform(-0.08, 0.08, size=(6, dimension))
    with scene(dimension=dimension, strains=strains, GRAVITY=[0, 0, 0], ENABLE_CONTACT=False) as (root, module):
        radius = module.RADIUS
        area = math.pi * radius**2
        inertia = math.pi * radius**4 / 4
        shear_modulus = module.YOUNG_MODULUS / (2 * (1 + module.POISSON_RATIO))
        diagonal = [shear_modulus * 2 * inertia,
                    module.YOUNG_MODULUS * inertia, module.YOUNG_MODULUS * inertia]
        if dimension == 6:
            diagonal += [module.YOUNG_MODULUS * area, shear_modulus * area, shear_modulus * area]
        stiffness = np.array(diagonal)
        h = module.LENGTH / module.ELEMENT_COUNT
        expected = sum(h / 6 * np.sum(stiffness * (a*a + a*b + b*b))
                       for a, b in zip(strains[:-1], strains[1:]))
        measured = root.rod.strain.material.elasticity.getPotentialEnergy()
        assert np.isclose(measured, expected, rtol=1e-11, atol=1e-13), (measured, expected)
        assert len(root.rod.strain.state.position.value) == module.ELEMENT_COUNT + 1
        return {"dimension": dimension, "measured": measured, "analytic": expected}


def constant_strain_limit(dimension):
    from scipy.linalg import expm
    value = np.array([0.21, -0.17, 0.33, 0.015, -0.01, 0.02][:dimension])
    with scene(dimension=dimension, strains=np.tile(value, (6, 1)),
               GRAVITY=[0, 0, 0], ENABLE_CONTACT=False) as (root, module):
        actual = frame_state(root).position.array().copy()
        x, y, z = value[:3]
        hat = np.zeros((4, 4))
        hat[:3, :3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        hat[:3, 3] = [1, 0, 0]
        if dimension == 6:
            hat[:3, 3] += value[3:]
        expected = np.array([expm(hat * coordinate) for coordinate in
                             np.linspace(0, module.LENGTH, len(actual))])
        position_error = float(np.max(np.abs(actual[:, :3] - expected[:, :3, 3])))
        rotation_error = float(np.max((Rotation.from_quat(actual[:, 3:]).inv()
            * Rotation.from_matrix(expected[:, :3, :3])).magnitude()))
        assert position_error < 1e-10 and rotation_error < 1e-10
        return {"dimension":dimension,"position_error":position_error,"rotation_error":rotation_error}


def assembled_material_solver(dimension):
    root = Sofa.Core.Node("assembledMaterial")
    root.dt = 0.01
    root.gravity = [0, 0, 0]
    root.addObject("RequiredPlugin", pluginName=[
        "Cosserat", "Sofa.Component.AnimationLoop", "Sofa.Component.StateContainer",
        "Sofa.Component.IntegrationScheme.Backward", "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mass", "Sofa.Component.MechanicalLoad",
    ])
    root.addObject("DefaultAnimationLoop")
    root.addObject("EulerImplicitIntegrationScheme", rayleighMass=0, rayleighStiffness=0)
    root.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
    template = f"Vec{dimension}d"
    state = root.addObject("MechanicalObject", template=template, name="state",
                           position=np.zeros((2, dimension)).tolist())
    root.addObject("UniformMass", template=template, totalMass=1.0)
    external = np.arange(1, 2 * dimension + 1).reshape(2, dimension) * 0.01
    root.addObject("ConstantForceField", template=template, indices=[0, 1], forces=external.tolist())
    samples = root.addChild("material")
    sampled = samples.addObject("MechanicalObject", template=template, name="state")
    offset = 1 / (2 * math.sqrt(3))
    samples.addObject("PiecewiseLinearStrainMapping", template=template,
                      input=state.getLinkPath(), output=sampled.getLinkPath(),
                      nodeAbscissa=[0, 1], sampleAbscissa=[0.5 - offset, 0.5 + offset])
    samples.addObject("BeamHookeLawForceField", template=template, length=[0.5, 0.5],
                      radius=0.01, youngModulus=2e6, poissonRatio=0.3)
    try:
        Sofa.Simulation.init(root)
        check_scene(root)
        inertia = math.pi * 0.01**4 / 4
        shear = 2e6 / 2.6
        diagonal = [2 * shear * inertia, 2e6 * inertia, 2e6 * inertia]
        if dimension == 6:
            area = math.pi * 0.01**2
            diagonal += [2e6 * area, shear * area, shear * area]
        material = np.diag(diagonal)
        stiffness = np.block([[material / 3, material / 6], [material / 6, material / 3]])
        expected = 0.01**2 * np.linalg.solve(np.eye(2 * dimension) * 0.5 + 0.01**2 * stiffness,
                                            external.ravel())
        Sofa.Simulation.animate(root, 0.01)
        check_scene(root)
        error = float(np.max(np.abs(state.position.array().ravel() - expected)))
        assert error < 1e-10, (error, state.position.array(), expected)
        return {"dimension": dimension, "maximum_position_error": error}
    finally:
        Sofa.Simulation.unload(root)


def integration_convergence(dimension):
    nodes = np.linspace(0, 0.5, 6)
    strains = np.column_stack([0.25 + nodes, -0.2 + nodes**2, 0.3 + 1.2 * nodes])
    if dimension == 6:
        strains = np.column_stack([strains, 0.01 + 0.02 * nodes, -0.01 * nodes, 0.015 * nodes])

    def rhs(s, flat):
        transform = flat.reshape(4, 4)
        q = np.array([np.interp(s, nodes, strains[:, axis]) for axis in range(dimension)])
        wx, wy, wz = q[:3]
        hat = np.zeros((4, 4))
        hat[:3, :3] = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
        hat[:3, 3] = [1, 0, 0]
        if dimension == 6:
            hat[:3, 3] += q[3:]
        return (transform @ hat).ravel()

    reference = np.eye(4)
    for a, b in zip(nodes[:-1], nodes[1:]):
        solution = solve_ivp(rhs, (a, b), reference.ravel(), method="DOP853", rtol=1e-12, atol=1e-14)
        assert solution.success
        reference = solution.y[:, -1].reshape(4, 4)
    errors = []
    for substeps in [2, 4, 8, 16]:
        with scene(dimension=dimension, substeps=substeps, strains=strains,
                   GRAVITY=[0, 0, 0], ENABLE_CONTACT=False) as (root, _):
            tip = frame_state(root).position.array()[-1]
            computed = np.eye(4)
            computed[:3, :3] = Rotation.from_quat(tip[3:]).as_matrix()
            computed[:3, 3] = tip[:3]
            errors.append(float(np.linalg.norm(computed - reference)))
    assert all(a > 3.5 * b for a, b in zip(errors[:-1], errors[1:])), errors
    return {"dimension": dimension, "substeps": [2, 4, 8, 16], "se3_errors": errors}


def tip_force_and_reset(model, dimension):
    with scene(model=model, dimension=dimension, GRAVITY=[0, 0, 0], ENABLE_CONTACT=False,
               TIP_FORCE=[0, -0.005, 0, 0, 0, 0]) as (root, _):
        initial = frame_state(root).position.array().copy()
        positions = animate(root, 150)
        tip_y = float(positions[-1, 1])
        assert tip_y < -1e-6, tip_y
        assert np.linalg.norm(positions[0, :3]) < 1e-10
        Sofa.Simulation.reset(root)
        reset_error = float(np.max(np.abs(frame_state(root).position.array() - initial)))
        assert reset_error < 1e-10, reset_error
        return {"model": model, "dimension": dimension, "solver": "SparseLDLSolver",
                "tip_y": tip_y, "reset_error": reset_error}


def contact_interaction(dimension):
    with scene(dimension=dimension, FLOOR_HEIGHT=-0.015, ENABLE_CONTACT=True) as (root, module):
        max_force = 0.0
        for _ in range(300):
            animate(root, 1)
            forces = np.asarray(root.constraints.constraintForces.value)
            if forces.size:
                max_force = max(max_force, float(np.max(np.abs(forces))))
        minimum = float(np.min(frame_state(root).position.array()[:, 1]))
        gap = minimum - module.FLOOR_HEIGHT - module.RADIUS
        assert max_force > 1e-8, "No contact constraint force was observed"
        assert gap > -0.004, gap
        return {"dimension": dimension, "minimum_surface_gap": gap, "maximum_constraint_force": max_force}


def cable_interaction(dimension):
    def add_cable(root, module):
        root.addObject("RequiredPlugin", pluginName=["SoftRobots"])
        count = module.ELEMENT_COUNT * module.INTEGRATION_SUBSTEPS + 1
        cable = root.rod.base.frames.addChild("cable")
        cable.addObject("MechanicalObject", template="Vec3d", name="state",
                         position=[[module.LENGTH * i / (count - 1), 0.015, 0] for i in range(count)])
        cable.addObject("RigidMapping", rigidIndexPerPoint=list(range(count)), globalToLocalCoords=True)
        cable.addObject("CableConstraint", name="actuator", indices=list(range(1, count)),
                         pullPoint=[0, 0.015, 0], value=[0.0003], valueType="displacement")
    with scene(dimension=dimension, GRAVITY=[0, 0, 0], ENABLE_CONTACT=False,
               before_init=add_cable) as (root, _):
        positions = animate(root, 120)
        tip_y = float(positions[-1, 1])
        assert abs(tip_y) > 1e-5, tip_y
        return {"dimension": dimension, "tip_y": tip_y}


def consistent_mass(dimension):
    def isolate_mass(root, module):
        root.rod.strain.material.removeObject(root.rod.strain.material.elasticity)
        ode = root.rod.getObject("EulerImplicitIntegrationScheme")
        ode.rayleighMass.value = 0
        ode.rayleighStiffness.value = 0

    velocity = np.random.default_rng(284 + dimension).normal(0, 0.1, (6, dimension))
    with scene(dimension=dimension, velocities=velocity, before_init=isolate_mass,
               GRAVITY=[0, 0, 0], ENABLE_CONTACT=False,
               TIP_FORCE=[1e-9, 0, 0, 0, 0, 0]) as (root, module):
        length, radius = module.LENGTH, module.RADIUS
        count = module.ELEMENT_COUNT * module.INTEGRATION_SUBSTEPS
        sections = np.linspace(0, length, count + 1)
        node_positions = np.linspace(0, length, module.ELEMENT_COUNT + 1)
        h = length / module.ELEMENT_COUNT
        midpoint_weights = np.zeros((count, len(node_positions)))
        for e, midpoint in enumerate((sections[:-1] + sections[1:]) / 2):
            left = min(int(midpoint / h), len(node_positions) - 2)
            t = (midpoint - node_positions[left]) / h
            midpoint_weights[e, left:left + 2] = [1 - t, t]
        offset = radius / math.sqrt(2)
        cross_section = np.array([[0, offset, 0], [0, -offset, 0],
                                  [0, 0, offset], [0, 0, -offset]])
        xis = [0.5 - 1 / (2 * math.sqrt(3)), 0.5 + 1 / (2 * math.sqrt(3))]
        weights = [0.5, 0.5]
        expected = np.zeros((velocity.size, velocity.size))
        kinetic = 0.0
        total = 0.0
        cross_inertia = np.zeros((3, 3))
        for k, (xi, weight) in enumerate(zip(xis, weights)):
            group = root.rod.base.massFrames.getChild(f"quadrature{k}")
            particle_mass = float(group.mass.vertexMass.value)
            assert np.isclose(particle_mass, module.TOTAL_MASS * weight / (4 * count))
            point_velocities = group.state.velocity.array().copy()
            kinetic += particle_mass * np.sum(point_velocities**2) / 2
            total += particle_mass * len(point_velocities)
            for e, (a, b) in enumerate(zip(sections[:-1], sections[1:])):
                s = a + (b - a) * xi
                covered = np.clip(s - sections[:-1], 0, np.diff(sections))
                A = covered @ midpoint_weights
                B = (covered * (s - sections[:-1] - covered / 2)) @ midpoint_weights
                for cross in cross_section:
                    J = np.zeros((3, velocity.size))
                    for node in range(len(node_positions)):
                        for axis in range(3):
                            J[:, dimension * node + axis] = (
                                B[node] * np.cross(np.eye(3)[axis], [1, 0, 0])
                                + A[node] * np.cross(np.eye(3)[axis], cross))
                        if dimension == 6:
                            J[:, dimension * node + 3:dimension * node + 6] = A[node] * np.eye(3)
                    expected += particle_mass * J.T @ J
                    cross_inertia += particle_mass * (np.dot(cross, cross) * np.eye(3)
                                                      - np.outer(cross, cross))
        assert np.isclose(total, module.TOTAL_MASS, rtol=1e-13)
        expected_inertia = module.TOTAL_MASS * np.diag([radius**2 / 2, radius**2 / 4, radius**2 / 4])
        assert np.allclose(cross_inertia, expected_inertia, rtol=1e-12, atol=1e-20)
        expected_energy = velocity.ravel() @ expected @ velocity.ravel() / 2
        assert np.isclose(kinetic, expected_energy, rtol=1e-11, atol=1e-16), (kinetic, expected_energy)
        # Remove initial velocity before the mass-only linear solve. The matrix
        # is assembled at the undeformed state and has M/dt as its free block.
        root.rod.strain.state.velocity.value = np.zeros_like(velocity).tolist()
        Sofa.Simulation.animate(root, float(root.dt.value))
        actual = root.rod.linearSolver.A().toarray()[6:, 6:] * float(root.dt.value)
        matrix_error = float(np.max(np.abs(actual - expected)))
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-14), matrix_error
        symmetry_error = float(np.max(np.abs(actual - actual.T)))
        minimum_eigenvalue = float(np.linalg.eigvalsh(actual).min())
        off_diagonal = actual - np.diag(np.diag(actual))
        assert symmetry_error < 1e-14 and minimum_eigenvalue > 0
        assert np.max(np.abs(off_diagonal)) > 0.1 * np.max(np.diag(actual))
        return {"dimension": dimension, "matrix_size": list(actual.shape),
                "total_mass": total, "cross_section_inertia": np.diag(cross_inertia).tolist(),
                "matrix_maximum_error": matrix_error, "symmetry_error": symmetry_error,
                "minimum_eigenvalue": minimum_eigenvalue,
                "largest_off_diagonal": float(np.max(np.abs(off_diagonal))),
                "kinetic_energy": float(kinetic), "analytic_kinetic_energy": float(expected_energy)}


def axial_extension():
    with scene(dimension=6, GRAVITY=[0, 0, 0], ENABLE_CONTACT=False,
               TIP_FORCE=[1, 0, 0, 0, 0, 0]) as (root, module):
        initial = frame_state(root).position.array().copy()
        positions = animate(root, 300)
        extension = float(positions[-1, 0] - initial[-1, 0])
        strains = root.rod.strain.state.position.array().copy()
        expected = module.LENGTH / (module.YOUNG_MODULUS * math.pi * module.RADIUS**2)
        assert extension > 1e-5 and np.all(strains[:, 3] > 0)
        assert np.isclose(extension, expected, rtol=0.01)
        return {"dimension": 6, "force_x": 1, "extension": extension,
                "analytic_static_extension": expected, "axial_strains": strains[:, 3].tolist()}


def geometry_jacobian_regression(model, dimension):
    count = 6 if model == "PLS" else 5
    rng = np.random.default_rng(912 + dimension)
    velocity = rng.normal(0, 0.2, (count, dimension))
    reference_strains = rng.normal(0, 1, (count, dimension))
    if dimension == 6:
        reference_strains[:, 3:] *= 0.03
    errors = []
    for scale in [0.0, 1e-10, 0.4, 4.0]:
        strains = reference_strains * scale
        with scene(model=model, dimension=dimension, strains=strains, velocities=velocity,
                   GRAVITY=[0, 0, 0], ENABLE_CONTACT=False) as (root, _):
            actual = np.concatenate([root.rod.base.massFrames.getChild(f"quadrature{i}")
                                     .state.velocity.array().copy() for i in range(2)])
        step = 1e-6
        positions = []
        for sign in [-1, 1]:
            with scene(model=model, dimension=dimension, strains=strains + sign * step * velocity,
                       GRAVITY=[0, 0, 0], ENABLE_CONTACT=False) as (root, _):
                positions.append(np.concatenate([
                    root.rod.base.massFrames.getChild(f"quadrature{i}").state.position.array().copy()
                    for i in range(2)]))
        finite_difference = (positions[1] - positions[0]) / (2 * step)
        error = float(np.max(np.abs(actual - finite_difference)))
        assert error < 2e-8, (model, dimension, scale, error)
        errors.append(error)
    return {"model": model, "dimension": dimension, "strain_scales": [0, 1e-10, 0.4, 4],
            "maximum_velocity_fd_errors": errors}


def sparse_output_frames():
    """Sparse output frames must preserve geometry, velocities and load transfer."""
    samples = np.array([0.0, 0.123, 0.37, 0.5])
    dense_samples = np.unique(np.concatenate([np.linspace(0, 0.5, 41), samples]))
    selected = np.searchsorted(dense_samples, samples)
    rng = np.random.default_rng(4185)
    strains = rng.normal(0, 0.4, (6, 6))
    strains[:, 3:] *= 0.03
    velocity = rng.normal(0, 0.2, (6, 6))
    loads = rng.normal(0, 0.02, (len(samples), 6))

    def configure(abscissa, force_indices=None):
        def setup(root, module):
            frames = root.rod.base.frames
            frames.removeObject(frames.getObject("ConstantForceField"))
            frames.state.position.value = [[s, 0, 0, 0, 0, 0, 1] for s in abscissa]
            frames.geometry.curv_abs_output.value = abscissa.tolist()
            if force_indices is not None:
                root.rod.strain.material.removeObject(root.rod.strain.material.elasticity)
                frames.addObject("ConstantForceField", template="Rigid3d",
                                 indices=force_indices.tolist(), forces=loads.tolist())
        return setup

    settings = dict(dimension=6, GRAVITY=[0, 0, 0], ENABLE_CONTACT=False,
                    ENABLE_INERTIA_BIAS=False, RAYLEIGH_MASS=0, RAYLEIGH_STIFFNESS=0)
    def mapped(abscissa, q):
        with scene(strains=q, velocities=velocity,
                   before_init=configure(abscissa), **settings) as (root, _):
            return (frame_state(root).position.array().copy(),
                    frame_state(root).velocity.array().copy())

    sparse, mapped_velocity = mapped(samples, strains)
    dense, dense_velocity = mapped(dense_samples, strains)
    position_error = float(np.max(np.abs(sparse - dense[selected])))
    velocity_error = float(np.max(np.abs(mapped_velocity - dense_velocity[selected])))
    assert position_error < 1e-12 and velocity_error < 1e-12, (position_error, velocity_error)
    step = 1e-6
    minus, _ = mapped(samples, strains - step * velocity)
    plus, _ = mapped(samples, strains + step * velocity)
    fd = np.column_stack([(plus[:, :3] - minus[:, :3]) / (2 * step),
        (Rotation.from_quat(plus[:, 3:]) * Rotation.from_quat(minus[:, 3:]).inv()).as_rotvec() / (2 * step)])
    fd_error = float(np.max(np.abs(mapped_velocity - fd)))
    assert fd_error < 2e-8, fd_error

    forces = []
    for abscissa, indices in [(samples, np.arange(len(samples))), (dense_samples, selected)]:
        with scene(strains=strains, TIME_STEP=1e-7,
                   before_init=configure(abscissa, indices), **settings) as (root, _):
            animate(root, 1)
            forces.append(root.rod.strain.state.force.array().copy())
    force_error = float(np.max(np.abs(forces[0] - forces[1])))
    assert force_error < 1e-11, force_error
    # The fixed base has zero virtual velocity; only strain work contributes.
    work_error = float(abs(np.sum(forces[0] * velocity) - np.sum(loads * mapped_velocity)))
    assert work_error < 1e-10, work_error
    return {"output_frames": len(samples), "integration_intervals": 40,
            "position_error": position_error, "velocity_error": velocity_error,
            "velocity_fd_error": fd_error, "force_transpose_error": force_error,
            "virtual_work_error": work_error}


def mass_convergence():
    length, radius, total_mass = 0.5, 0.01, 0.015
    nodes = np.linspace(0, length, 6)
    cross_offset = radius / math.sqrt(2)
    crosses = np.array([[0, cross_offset, 0], [0, -cross_offset, 0],
                        [0, 0, cross_offset], [0, 0, -cross_offset]])
    reference = np.zeros((36, 36))
    # At a straight reference, the continuous PLS velocity Jacobian is cubic
    # within a material element. Four Gauss points integrate its quadratic
    # kinetic energy exactly. This high-order rule is a validation reference;
    # the delivered scene keeps the user's two-point mass rule.
    xi4, w4 = np.polynomial.legendre.leggauss(4)
    xi2, w2 = np.polynomial.legendre.leggauss(2)
    for left, right in zip(nodes[:-1], nodes[1:]):
        for xi, weight in zip(xi4, w4):
            s = (left + right) / 2 + (right - left) * xi / 2
            A, B = np.zeros(6), np.zeros(6)
            for e, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
                upper = min(s, b)
                if upper <= a:
                    continue
                for uxi, uw in zip(xi2, w2):
                    u = (a + upper) / 2 + (upper - a) * uxi / 2
                    t = (u - a) / (b - a)
                    shape = np.zeros(6)
                    shape[e:e + 2] = [1 - t, t]
                    A += (upper - a) * uw / 2 * shape
                    B += (upper - a) * uw / 2 * (s - u) * shape
            particle_mass = total_mass / length * (right - left) * weight / 8
            for cross in crosses:
                J = np.zeros((3, 36))
                for node in range(6):
                    for axis in range(3):
                        J[:, 6 * node + axis] = (B[node] * np.cross(np.eye(3)[axis], [1, 0, 0])
                                                + A[node] * np.cross(np.eye(3)[axis], cross))
                    J[:, 6 * node + 3:6 * node + 6] = A[node] * np.eye(3)
                reference += particle_mass * J.T @ J

    def isolate_mass(root, module):
        root.rod.strain.material.removeObject(root.rod.strain.material.elasticity)
        ode = root.rod.getObject("EulerImplicitIntegrationScheme")
        ode.rayleighMass.value = 0
        ode.rayleighStiffness.value = 0

    errors = []
    for substeps in [2, 4, 8, 16]:
        with scene(dimension=6, substeps=substeps, before_init=isolate_mass,
                   GRAVITY=[0, 0, 0], ENABLE_CONTACT=False,
                   TIP_FORCE=[1e-9, 0, 0, 0, 0, 0]) as (root, _):
            Sofa.Simulation.animate(root, float(root.dt.value))
            actual = root.rod.linearSolver.A().toarray()[6:, 6:] * float(root.dt.value)
            errors.append(float(np.linalg.norm(actual - reference) / np.linalg.norm(reference)))
    assert all(a > 3.5 * b for a, b in zip(errors[:-1], errors[1:])), errors
    return {"dimension": 6, "mass_gauss_points_per_subinterval": 2,
            "substeps": [2, 4, 8, 16], "relative_matrix_errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["all", "mechanics", "mass", "regression", "dynamics", "contact", "cable"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    def run(name, function, *arguments):
        results.setdefault(name, []).append(function(*arguments))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print("PASS", name, results[name][-1], flush=True)
    for dimension in [6]:
        if args.case in ("all", "mechanics"):
            run("material_energy", material_energy, dimension)
            run("assembled_material_solver", assembled_material_solver, dimension)
            run("constant_strain_limit", constant_strain_limit, dimension)
            run("integration_convergence", integration_convergence, dimension)
        if args.case in ("all", "regression"):
            for model in ["PLS"]:
                run("geometry_jacobian_regression", geometry_jacobian_regression, model, dimension)
            run("sparse_output_frames", sparse_output_frames)
        if args.case in ("all", "mass"):
            run("consistent_mass", consistent_mass, dimension)
            if dimension == 6:
                run("axial_extension", axial_extension)
                run("mass_convergence", mass_convergence)
        if args.case in ("all", "dynamics"):
            for model in ["PLS"]:
                run("tip_force_and_reset", tip_force_and_reset, model, dimension)
        if args.case in ("all", "contact"):
            run("contact_interaction", contact_interaction, dimension)
        if args.case in ("all", "cable"):
            run("cable_interaction", cable_interaction, dimension)


if __name__ == "__main__":
    main()
