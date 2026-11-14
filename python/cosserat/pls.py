"""PLS strain, frame and mass helpers for circular Cosserat rods.

The examples use these functions inside the upstream getting-started scene
layout. They do not select solvers, constrain the base or create contacts.
"""
from dataclasses import dataclass
import math
from types import SimpleNamespace


@dataclass(frozen=True)
class RodParameters:
    length: float = 0.5
    elements: int = 10
    substeps: int = 3
    radius: float = 0.01
    young_modulus: float = 2e6
    poisson_ratio: float = 0.3
    mass: float = 0.015
    model: str = "PLS"
    dimension: int = 6
    rayleigh_stiffness: float = 0.0
    rigidities: dict = None


def add_strain(solver, p, *, node_name="strain", state_name="state",
               integration_state_name="state"):
    """Nodal strain, two-point material quadrature and midpoint geometry samples."""
    if p.model not in ("PLS", "PCS") or p.dimension not in (3, 6):
        raise ValueError("Choose PLS/PCS with Vec3 or Vec6 strain")
    if p.elements < 1 or p.substeps < 1 or p.length <= 0:
        raise ValueError("Length and discretization counts must be positive")
    h = p.length / p.elements
    nodes = [i * h for i in range(p.elements + 1)]
    template = f"Vec{p.dimension}d"
    strain = solver.addChild(node_name)
    count = len(nodes) if p.model == "PLS" else p.elements
    state = strain.addObject("MechanicalObject", template=template, name=state_name,
                             position=[[0.0] * p.dimension for _ in range(count)])
    if p.model == "PLS":
        material = strain.addChild("material")
        sample = material.addObject("MechanicalObject", template=template, name="state")
        gauss = [nodes[e] + h * (0.5 + sign / (2 * math.sqrt(3)))
                 for e in range(p.elements) for sign in (-1, 1)]
        material.addObject("PiecewiseLinearStrainMapping", template=template,
                            input=state.getLinkPath(), output=sample.getLinkPath(),
                            nodeAbscissa=nodes, sampleAbscissa=gauss)
        lengths = [h / 2] * len(gauss)
        integration = strain.addChild("integration")
        integration_state = integration.addObject("MechanicalObject", template=template, name=integration_state_name)
        n = p.elements * p.substeps
        sections = [i * p.length / n for i in range(n + 1)]
        integration.addObject("PiecewiseLinearStrainMapping", template=template,
                              input=state.getLinkPath(), output=integration_state.getLinkPath(),
                              nodeAbscissa=nodes,
                              sampleAbscissa=[(a + b) / 2 for a, b in zip(sections[:-1], sections[1:])])
    else:
        material, integration, integration_state = strain, strain, state
        sections, lengths = nodes, [h] * p.elements
    material_options = dict(youngModulus=p.young_modulus, poissonRatio=p.poisson_ratio)
    if p.rigidities is not None:
        material_options.update(useInertiaParams=True, **p.rigidities)
    material.addObject("BeamHookeLawForceField", name="elasticity", template=template,
                       crossSectionShape="circular", length=lengths, radius=p.radius,
                       rayleighStiffness=p.rayleigh_stiffness, **material_options)
    n = p.elements * p.substeps
    return SimpleNamespace(node=strain, nodal_state=state, integration=integration, state=integration_state,
                           sections=sections, nodes=nodes, abscissa=[i * p.length / n for i in range(n + 1)])


def add_frames(base, strain, p, *, base_state=None, name="frames", state_name="state",
               mapping_name="geometry", positions=None, show_object=True):
    base_state = base.state if base_state is None else base_state
    frames = base.addChild(name)
    strain.integration.addChild(frames)
    state = frames.addObject("MechanicalObject", template="Rigid3d", name=state_name,
                              position=(positions if positions is not None else
                                        [[s, 0, 0, 0, 0, 0, 1] for s in strain.abscissa]),
                              showObject=show_object, showObjectScale=p.radius * 0.65)
    frames.addObject("Strain2FramesCosseratMapping", name=mapping_name, plsKinematics=p.model == "PLS",
                      input1=strain.state.getLinkPath(), input2=base_state.getLinkPath(),
                      output=state.getLinkPath(), curv_abs_input=strain.sections,
                      curv_abs_output=strain.abscissa, radius=p.radius, nonColored=True)
    return frames


def add_quadrature_mass(base, strain, p, *, inertia_bias, base_state=None):
    """Two axial Gauss points per interval and four per cross section.

    Internal quadrature states use the native NoPicking tag: hidden mechanical
    states are otherwise still eligible for native mouse picking.
    """
    if p.mass < 0:
        raise ValueError("Mass must be nonnegative")
    if p.mass == 0:
        return None
    base_state = base.state if base_state is None else base_state
    gauss = [0.5 - 1 / (2 * math.sqrt(3)), 0.5 + 1 / (2 * math.sqrt(3))]
    abscissa = [0.0] + [a + (b - a) * xi
                        for a, b in zip(strain.sections[:-1], strain.sections[1:]) for xi in gauss]
    frames = base.addChild("massFrames")
    strain.integration.addChild(frames)
    frames.addObject("VisualStyle", displayFlags="hideMappings hideMechanicalMappings hideBehaviorModels")
    state = frames.addObject("MechanicalObject", template="Rigid3d", name="state",
                              position=[[s, 0, 0, 0, 0, 0, 1] for s in abscissa], tags="NoPicking")
    frames.addObject("Strain2FramesCosseratMapping", name="geometry", plsKinematics=p.model == "PLS",
                      input1=strain.state.getLinkPath(), input2=base_state.getLinkPath(),
                      output=state.getLinkPath(), curv_abs_input=strain.sections,
                      curv_abs_output=abscissa, radius=p.radius)
    offset = p.radius / math.sqrt(2)
    local_points = [[0, offset, 0], [0, -offset, 0], [0, 0, offset], [0, 0, -offset]]
    intervals = len(strain.sections) - 1
    for k in range(2):
        group = frames.addChild(f"quadrature{k}")
        points = group.addObject("MechanicalObject", template="Vec3d", name="state",
                                  position=local_points * intervals, tags="NoPicking")
        group.addObject("UniformMass", template="Vec3d", name="mass", vertexMass=p.mass / (8 * intervals))
        group.addObject("RigidMapping", name="sectionPoints", input=state.getLinkPath(),
                        output=points.getLinkPath(), rigidIndexPerPoint=[1 + 2 * e + k
                            for e in range(intervals) for _ in local_points], globalToLocalCoords=False)
        if p.model == "PLS" and inertia_bias:
            group.addObject("PLSInertiaForceField", name="inertiaBias",
                            cosseratMapping=frames.geometry.getLinkPath(),
                            rigidMapping=group.sectionPoints.getLinkPath(), mass=group.mass.getLinkPath())

    return frames
