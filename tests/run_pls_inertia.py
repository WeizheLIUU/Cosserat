"""Independent convective-inertia validation against positions and mass derivatives.

Run serially with this workspace's SofaPython3 on PYTHONPATH and SOFA_ROOT set.
Runtime implementation uses analytic motion jets; finite differences occur here only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run_pls_integration import scene, animate


def isolate(root, module):
    root.rod.strain.material.removeObject(root.rod.strain.material.elasticity)
    ode = root.rod.getObject("EulerImplicitIntegrationScheme")
    ode.rayleighMass.value = 0
    ode.rayleighStiffness.value = 0


def states(q, v=None, enabled=True, mass=.015, dimension=6):
    return scene(dimension=dimension, strains=q, velocities=v, GRAVITY=[0, 0, 0],
                 ENABLE_CONTACT=False, TIME_STEP=1e-9, ENABLE_INERTIA_BIAS=enabled,
                 TOTAL_MASS=mass, before_init=isolate)


def groups(root):
    return [root.rod.base.massFrames.getChild(f"quadrature{k}") for k in [0, 1]]


def sample_points(q):
    with states(q, enabled=False) as (root, _):
        return np.concatenate([g.state.position.array().copy() for g in groups(root)])


def sample_forces(q, v, mass=.015, dimension=6):
    with states(q, v, mass=mass, dimension=dimension) as (root, _):
        animate(root, 1)
        return (root.rod.strain.state.force.array().copy().reshape(-1),
                np.concatenate([g.inertiaBias.biasForces.array().copy() for g in groups(root)]),
                [int(g.inertiaBias.motionBuilds.value) for g in groups(root)])


def mass_matrix(q):
    with states(q, enabled=False) as (root, _):
        animate(root, 1)
        return root.rod.linearSolver.A().toarray()[6:, 6:] * float(root.dt.value)


def math_checks():
    rng = np.random.default_rng(1953)
    q = rng.normal(size=(6, 6)) * .7
    q[:, 3:] *= .05
    v = rng.normal(size=(6, 6)) * .8
    v[:, 3:] *= .07
    x = sample_points(q)
    epsilon = 1e-3
    acceleration = (sample_points(q + epsilon*v) - 2*x
                    + sample_points(q - epsilon*v)) / epsilon**2
    columns = []
    for i in range(q.size):
        delta = np.zeros_like(q)
        delta.flat[i] = 1e-6
        columns.append(((sample_points(q + delta) - sample_points(q - delta)) / 2e-6).reshape(-1))
    jacobian = np.column_stack(columns)
    mass = .015 / len(x)
    force, bias, counts = sample_forces(q, v)
    expected = -jacobian.T @ (mass * acceleration.reshape(-1))
    expected_points = -mass * acceleration
    result = {"case": "positions_and_christoffel",
        "position_second_difference_relative_error": float(np.linalg.norm(bias - expected_points) / np.linalg.norm(expected_points)),
        "nodal_inertia_relative_error": float(np.linalg.norm(force - expected) / np.linalg.norm(expected)),
        "nodal_max_abs_error": float(np.max(np.abs(force - expected))),
        "shared_motion_build_counts": counts,
        "force_norm": float(np.linalg.norm(force))}
    assert result["position_second_difference_relative_error"] < 2e-5, result
    assert result["nodal_inertia_relative_error"] < 2e-5, result
    assert counts == [1, 1], counts
    zero, _, _ = sample_forces(q, np.zeros_like(v))
    negative, _, _ = sample_forces(q, -v)
    doubled, _, _ = sample_forces(q, 2*v)
    mass_doubled, _, _ = sample_forces(q, v, mass=.03)
    assert np.max(np.abs(zero)) == 0
    assert np.allclose(force, negative, rtol=1e-10, atol=1e-14)
    assert np.allclose(doubled, 4*force, rtol=1e-10, atol=1e-14)
    assert np.allclose(mass_doubled, 2*force, rtol=1e-10, atol=1e-14)
    result.update(zero_velocity_max=float(np.max(np.abs(zero))),
        even_velocity_error=float(np.max(np.abs(force-negative))),
        quadratic_velocity_error=float(np.max(np.abs(doubled-4*force))),
        mass_scaling_error=float(np.max(np.abs(mass_doubled-2*force))))
    scaling_errors = []
    for scale in [1e-6, 10., 1e3]:
        scaled, _, _ = sample_forces(q, scale*v)
        error = float(np.linalg.norm(scaled-scale**2*force)/(scale**2*np.linalg.norm(force)))
        assert error < 1e-10, error
        scaling_errors.append({"scale": scale, "relative_error": error})
    result["wide_velocity_scaling"] = scaling_errors
    derivatives = []
    for i in range(q.size):
        delta = np.zeros_like(q)
        delta.flat[i] = 1e-4
        derivatives.append((mass_matrix(q+delta) - mass_matrix(q-delta)) / 2e-4)
    velocity = v.reshape(-1)
    mass_dot = sum(velocity[i]*derivatives[i] for i in range(q.size))
    gradient = np.array([.5*velocity @ d @ velocity for d in derivatives])
    christoffel = mass_dot @ velocity - gradient
    result["mass_christoffel_relative_error"] = float(np.linalg.norm(-force-christoffel) / np.linalg.norm(christoffel))
    result["kinetic_power_identity_error"] = float(abs(velocity @ (-force) - .5*velocity @ mass_dot @ velocity))
    assert result["mass_christoffel_relative_error"] < 2e-5, result
    q[:, 3:] = 0
    v[:, 3:] = 0
    f6, b6, _ = sample_forces(q, v)
    f3, b3, _ = sample_forces(q[:, :3], v[:, :3], dimension=3)
    result["vec3_limit_error"] = float(max(np.max(np.abs(f6.reshape(6, 6)[:, :3].reshape(-1)-f3)), np.max(np.abs(b6-b3))))
    assert result["vec3_limit_error"] < 1e-13, result
    return result


def base_checks():
    q = np.random.default_rng(33).normal(size=(6, 6)) * .3
    q[:, 3:] *= .02
    velocity = np.random.default_rng(34).normal(size=(6, 6)) * .5
    velocity[:, 3:] *= .04
    omega = np.array([.6, -.4, .8])
    center = np.array([.1, -.07, .12])
    rotation = Rotation.from_rotvec([.2, -.3, .1])
    results = []
    for deform in [False, True]:
        for linear in [np.zeros(3), np.array([.13, -.21, .07])]:
            def configure(root, module):
                isolate(root, module)
                root.rod.base.state.position.value = [[*center, *rotation.as_quat()]]
            with scene(dimension=6, strains=q, velocities=(velocity if deform else np.zeros_like(q)),
                       GRAVITY=[0, 0, 0], ENABLE_CONTACT=False, TIME_STEP=1e-9,
                       before_init=configure) as (root, _):
                positions = np.concatenate([g.state.position.array().copy() for g in groups(root)])
                root.rod.base.state.velocity.value = [[*linear, *omega]]
                animate(root, 1)
                actual = np.concatenate([g.inertiaBias.biasForces.array().copy() for g in groups(root)])
            if not deform:
                expected_acceleration = np.cross(omega, np.cross(omega, positions-center))
            else:
                # Independent manifold path: R(t)=Exp(t*omega)R0, p(t)=p0+t*u,
                # nodal q(t)=q+t*v. Includes mixed base/strain Coriolis terms.
                def sample(t):
                    def pose(root, module):
                        isolate(root, module)
                        quat = (Rotation.from_rotvec(t*omega)*rotation).as_quat()
                        root.rod.base.state.position.value = [[*(center+t*linear), *quat]]
                    with scene(dimension=6, strains=q+t*velocity, GRAVITY=[0, 0, 0],
                               ENABLE_CONTACT=False, ENABLE_INERTIA_BIAS=False,
                               before_init=pose) as (r, _):
                        return np.concatenate([g.state.position.array().copy() for g in groups(r)])
                epsilon = 1e-3
                expected_acceleration = (sample(epsilon)-2*positions+sample(-epsilon))/epsilon**2
            expected = -(.015/len(positions))*expected_acceleration
            error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            assert error < (2e-6 if deform else 1e-10), error
            results.append({"deforming": deform, "base_translation_velocity": linear.tolist(), "relative_error": error})
    return {"case": "moving_base_and_cross_terms", "results": results}


def energy_checks():
    q = np.zeros((6, 6)); q[5, 2] = 20
    velocity = np.zeros_like(q); velocity[5, 2] = 30
    def isolated_mode(root, module):
        isolate(root, module)
        root.rod.strain.addObject("FixedProjectiveConstraint", indices=list(range(5)))
        root.rod.strain.addObject("PartialFixedProjectiveConstraint", indices=[5],
                                 fixedDirections=[1, 1, 0, 1, 1, 1])
    delta = np.zeros_like(q); delta[5, 2] = 1e-5
    derivative = (sample_points(q+delta)-sample_points(q-delta))/2e-5
    mass = .015/len(derivative)
    initial_energy = .5*mass*np.sum((derivative*30)**2)
    results = []
    for enabled in [False, True]:
        for dt in [.002, .001, .0005]:
            with scene(dimension=6, strains=q, velocities=velocity, GRAVITY=[0, 0, 0],
                       ENABLE_CONTACT=False, TIME_STEP=dt, ENABLE_INERTIA_BIAS=enabled,
                       before_init=isolated_mode) as (root, _):
                energy = []
                for step in range(round(.1/dt)):
                    animate(root, 1)
                    cartesian_velocity = np.concatenate([g.state.velocity.array().copy() for g in groups(root)])
                    energy.append(.5*mass*np.sum(cartesian_velocity**2))
                results.append({"enabled": enabled, "dt": dt, "initial_energy": float(initial_energy),
                    "final_energy": float(energy[-1]),
                    "relative_drift": float((energy[-1]-initial_energy)/initial_energy),
                    "maximum_relative_drift": float(np.max(np.abs(np.array(energy)/initial_energy-1))),
                    "final_q": float(root.rod.strain.state.position.array()[5, 2]),
                    "final_velocity": float(root.rod.strain.state.velocity.array()[5, 2])})
    missing = [r["maximum_relative_drift"] for r in results if not r["enabled"]]
    complete = [r["maximum_relative_drift"] for r in results if r["enabled"]]
    assert min(missing) > .005, results
    assert complete[0] < 5e-5, results
    assert all(.45 < complete[i+1]/complete[i] < .55 for i in [0, 1]), results
    return {"case": "unforced_kinetic_energy_and_step_convergence", "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["math", "base", "energy", "all"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for name, check in [("math", math_checks), ("base", base_checks), ("energy", energy_checks)]:
        if args.case in (name, "all"):
            result = check(); results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2)+"\n")
            print("PASS", result, flush=True)


if __name__ == "__main__":
    main()
