"""Segmented tube visuals mapped to existing rod frames; no mechanical state."""
import math
import Sofa.Core


class _FrameVisibility(Sofa.Core.Controller):
    """Follow the native root display flag, including while animation is paused."""

    def __init__(self, style, state, **kwargs):
        super().__init__(**kwargs)
        self.style = style
        self.state = state

    def draw(self, visual_params):
        flags = set(self.style.displayFlags.getValueString().split())
        visible = False
        for group in ("BehaviorModels", "Behavior", "All", "Root"):
            if "show" + group in flags or "hide" + group in flags:
                visible = "show" + group in flags
                break
        if bool(self.state.showObject.value) != visible:
            self.state.showObject = visible


def add_segmented_visual(frames, abscissa, boundaries, radius_at, *, colors,
                         ring_points=16, show_boundaries=True, frame_state=None):
    """Color material elements, retaining all internal geometry samples.

    Element boundaries must coincide with output frames. Duplicate reference
    interface frames are coalesced for drawing only. No DOF or mass is added.
    """
    if ring_points < 3 or not colors:
        raise ValueError("At least three ring vertices and one color are required")
    frame_state = frames.state if frame_state is None else frame_state
    root = frames.getRoot()
    visibility = root.addObject(_FrameVisibility(
        root.VisualStyle, frame_state,
        name="frameVisibility_" + frame_state.getPathName().strip("/").replace("/", "_")))
    visibility.draw(None)
    unique = []
    for index, s in enumerate(abscissa):
        if not unique or abs(s - unique[-1][1]) > 1e-12:
            unique.append((index, s))
    boundary_frames = []
    for s in boundaries:
        index, actual = min(unique, key=lambda pair: abs(pair[1] - s))
        if abs(actual - s) > 1e-10:
            raise ValueError("Visual element boundaries must coincide with output frames")
        boundary_frames.append(index)
    visual = frames.addChild("visual")

    def model(node, vertices, triangles, indices, color):
        surface = node.addObject("OglModel", name="model", position=vertices,
                                 triangles=triangles, color=color)
        node.addObject("RigidMapping", input=frame_state.getLinkPath(),
                       output=surface.getLinkPath(), rigidIndexPerPoint=indices,
                       globalToLocalCoords=False, mapForces=False,
                       mapConstraints=False, mapMasses=False)

    for e, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        samples = [(index, s) for index, s in unique if left - 1e-12 <= s <= right + 1e-12]
        vertices, triangles, indices = [], [], []
        for ring, (index, s) in enumerate(samples):
            radius = radius_at(s)
            for j in range(ring_points):
                theta = 2 * math.pi * j / ring_points
                vertices.append([0, radius * math.cos(theta), radius * math.sin(theta)])
                indices.append(index)
                if ring:
                    a = (ring - 1) * ring_points + j
                    b = (ring - 1) * ring_points + (j + 1) % ring_points
                    c = ring * ring_points + j
                    d = ring * ring_points + (j + 1) % ring_points
                    triangles.extend([[a, b, c], [b, d, c]])
        # Only the two physical ends are capped; internal elements meet exactly.
        for end, enabled in [(0, e == 0), (len(samples) - 1, e == len(boundaries) - 2)]:
            if enabled:
                center = len(vertices)
                vertices.append([0, 0, 0]); indices.append(samples[end][0])
                for j in range(ring_points):
                    a, b = end * ring_points + j, end * ring_points + (j + 1) % ring_points
                    triangles.append([center, b, a] if end == 0 else [center, a, b])
        model(visual.addChild(f"element{e:02d}"), vertices, triangles, indices, colors[e % len(colors)])

    if show_boundaries:
        vertices, triangles, indices = [], [], []
        for index, s in zip(boundary_frames, boundaries):
            radius = radius_at(s)
            start = len(vertices)
            for x in [-0.03 * radius, 0.03 * radius]:
                for j in range(ring_points):
                    theta = 2 * math.pi * j / ring_points
                    vertices.append([x, 1.015 * radius * math.cos(theta),
                                     1.015 * radius * math.sin(theta)])
                    indices.append(index)
            for j in range(ring_points):
                a, b = start + j, start + (j + 1) % ring_points
                c, d = a + ring_points, b + ring_points
                triangles.extend([[a, b, c], [b, d, c]])
        model(visual.addChild("boundaries"), vertices, triangles, indices, [0.12, 0.16, 0.20, 1.0])
    return visual
