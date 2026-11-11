"""! @brief Extract robot model data from a MuJoCo XML (MJCF) file.

The MJCF file is compiled with the `mujoco` package (which resolves
<include> files, defaults and compiler options such as angle units) and the
compiled model is converted into a URDF object (from urdf_parser_py). This
allows RobotModel to consume .xml (MJCF) files in exactly the same way as
.urdf files, e.g. OptimizationBuilder works unchanged.

The conversion follows the same conventions as generating a URDF from MJCF:

- each MJCF <body> becomes a URDF <link> (the worldbody becomes the root
  link, usually named "world"),
- each MJCF <joint> becomes a URDF joint whose child is a dummy link named
  "{joint_name}_jointbody", connected to the body link with a fixed
  "{joint_name}_offset" joint (hinge -> revolute, slide -> prismatic,
  freejoint -> floating),
- bodies without joints are attached to their parent with a fixed joint
  named "{parent_name}2{child_name}_fixed".
"""

import os
import re

import numpy as np

from urdf_parser_py.urdf import (
    URDF,
    Link,
    Joint,
    Pose,
    JointLimit,
    Inertial,
    Inertia,
    Visual,
    Mesh,
    Box,
    Sphere,
    Cylinder,
    Material,
    Color,
)

# Default actuator limits written into URDF joint limits. MuJoCo joints do
# not carry effort/velocity limits, so these defaults are used (they match
# the values typically used when exporting MJCF to URDF).
DEFAULT_EFFORT_LIMIT = 100.0
DEFAULT_VELOCITY_LIMIT = 100.0

# Mass/inertia of the dummy "{joint_name}_jointbody" links.
DUMMY_LINK_MASS = 0.001
DUMMY_LINK_INERTIA = DUMMY_LINK_MASS * (0.001**2)


def _quat2rpy(quat_wxyz) -> np.ndarray:
    """! Convert a MuJoCo quaternion [w, x, y, z] to roll-pitch-yaw angles.

    The returned rpy matches the convention used by optas.spatialmath.rpy2r,
    i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll) with rpy = [roll, pitch, yaw].

    @param quat_wxyz Quaternion in MuJoCo order [w, x, y, z].
    @return Array [roll, pitch, yaw] in radians.
    """
    w, x, y, z = [float(v) for v in quat_wxyz]
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll), handle the gimbal-lock cases
    # (|R[2, 0]| = 1, pitch = +/- pi/2) where roll/yaw are coupled
    if R[2, 0] < -1.0 + 1e-12:  # pitch = +pi/2
        pitch = np.pi / 2.0
        roll = np.arctan2(R[0, 1], R[0, 2])
        yaw = 0.0
    elif R[2, 0] > 1.0 - 1e-12:  # pitch = -pi/2
        pitch = -np.pi / 2.0
        roll = np.arctan2(-R[0, 1], -R[0, 2])
        yaw = 0.0
    else:
        pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def _quat2r(quat_wxyz) -> np.ndarray:
    """! Convert a MuJoCo quaternion [w, x, y, z] to a rotation matrix."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def is_mjcf_string(xml_string: str) -> bool:
    """! Detect whether an XML string is a MuJoCo (MJCF) model.

    @param xml_string XML content as a string.
    @return True when the root element is <mujoco>, False when <robot> (URDF).
    """
    match = re.search(r"<\s*([A-Za-z_][\w.-]*)", xml_string.lstrip())
    if match is None:
        return False
    return match.group(1).lower() == "mujoco"


def _spec_and_model(xml_filename: str):
    """! Load a MJCF file with the mujoco package.

    @param xml_filename Path to the .xml (MJCF) file.
    @return Tuple (spec, model, mesh_files, mesh_scales) where spec is the
            mujoco.MjSpec, model the compiled mujoco.MjModel, mesh_files a
            dict mapping mesh names to asset file paths (relative to the
            directory of xml_filename, with the compiler meshdir folded in)
            and mesh_scales a dict mapping mesh names to their scale (None
            when unit scale).
    """
    import mujoco  # lazy import, only needed for .xml files

    spec = mujoco.MjSpec.from_file(str(xml_filename))
    model = spec.compile()

    base_dir = os.path.dirname(os.path.abspath(str(xml_filename)))
    try:
        meshdir = spec.compiler.meshdir or ""
    except Exception:
        meshdir = ""

    mesh_files = {}
    mesh_scales = {}
    for mesh in spec.meshes:
        if mesh.name is None or mesh.file is None:
            continue
        file = mesh.file
        if meshdir:
            if os.path.isabs(meshdir):
                candidate = os.path.join(meshdir, file)
            else:
                candidate = os.path.join(meshdir, file)
                # mujoco resolves relative paths against the model directory;
                # fall back to the plain file if the meshdir variant is missing
                if not os.path.exists(os.path.join(base_dir, candidate)) and os.path.exists(
                    os.path.join(base_dir, file)
                ):
                    candidate = file
        else:
            candidate = file
        mesh_files[mesh.name] = candidate.replace(os.sep, "/")
        scale_attr = getattr(mesh, "scale", None)
        scale = np.asarray(scale_attr if scale_attr is not None else [1.0, 1.0, 1.0], dtype=float)
        mesh_scales[mesh.name] = None if np.allclose(scale, 1.0) else [float(s) for s in scale]
    return spec, model, mesh_files, mesh_scales


def _body_name(model, mujoco, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if name is None:
        name = f"body_{body_id}"
    return name


def _joint_name(model, mujoco, jnt_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    if name is None:
        name = f"joint_{jnt_id}"
    return name


def _make_inertial(mass, ipos, inertia_diag, iquat) -> Inertial:
    """! Build a URDF Inertial from compiled MuJoCo body inertial data.

    MuJoCo stores the inertia as a diagonal matrix in the frame given by
    iquat; it is rotated into the body (link) frame here.
    """
    R = _quat2r(iquat)
    I = R @ np.diag(np.asarray(inertia_diag, dtype=float)) @ R.T
    inertia = Inertia(
        ixx=I[0, 0], ixy=I[0, 1], ixz=I[0, 2],
        iyy=I[1, 1], iyz=I[1, 2], izz=I[2, 2],
    )
    origin = Pose(xyz=list(np.asarray(ipos, dtype=float)), rpy=list(_quat2rpy(iquat)))
    return Inertial(mass=float(mass), inertia=inertia, origin=origin)


def _make_visuals(model, mujoco, body_id: int, mesh_files: dict, mesh_scales: dict) -> list:
    """! Build URDF visuals for the geoms of a body.

    Mesh geoms become urdf.Mesh visuals (filename relative to the MJCF file
    directory), primitive geoms (box/sphere/cylinder/capsule) become the
    corresponding urdf primitive visuals. Planes, heightfields and ellipsoids
    are skipped as they are not link geometry.
    """
    visuals = []
    for k in range(model.body_geomnum[body_id]):
        geom_id = model.body_geomadr[body_id] + k
        # Skip visual-only, non-colliding geoms (contype=0 and conaffinity=0,
        # e.g. "goal"/"pred" helper bodies): they are not collision geometry
        # and should not end up in convex decompositions.
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        geom_type = int(model.geom_type[geom_id])
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is None:
            geom_name = f"geom_{geom_id}"
        geom_origin = Pose(
            xyz=list(model.geom_pos[geom_id]),
            rpy=list(_quat2rpy(model.geom_quat[geom_id])),
        )
        geom_size = np.asarray(model.geom_size[geom_id], dtype=float)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            if mesh_name is None:
                mesh_name = f"mesh_{mesh_id}"
            geometry = Mesh(
                filename=mesh_files.get(mesh_name, mesh_name),
                scale=mesh_scales.get(mesh_name),
            )
            # Attach the compiled mesh data. MuJoCo re-centers/re-orients mesh
            # vertices at compile time, so the raw asset file does not contain
            # the geometry used by the simulator; the compiled vertices/faces
            # (mesh_vert/mesh_face) do. They are stored on the Mesh object as
            # mjcf_vertices/mjcf_faces so consumers can build the exact
            # simulated geometry without re-loading the asset file.
            va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            geometry.mjcf_vertices = np.array(model.mesh_vert[va:va + vn], dtype=float)
            geometry.mjcf_faces = np.array(model.mesh_face[fa:fa + fn], dtype=int)
            visual_name = mesh_name
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            # MJCF box size is half-extents, URDF box size is full extents
            geometry = Box(size=[float(2.0 * s) for s in geom_size[:3]])
            visual_name = geom_name
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            geometry = Sphere(radius=float(geom_size[0]))
            visual_name = geom_name
        elif geom_type in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
            # URDF has no capsule; approximate with a cylinder of same radius
            # and length (MJCF size is radius, half-length)
            geometry = Cylinder(radius=float(geom_size[0]), length=float(2.0 * geom_size[1]))
            visual_name = geom_name
        else:
            continue  # plane, hfield, ellipsoid, sdf: not link geometry

        mat_id = model.geom_matid[geom_id]
        if mat_id != -1:
            rgba = [float(c) for c in model.mat_rgba[mat_id]]
            mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id)
        else:
            rgba = [float(c) for c in model.geom_rgba[geom_id]]
            mat_name = None
        material = Material(name=mat_name, color=Color(rgba))
        visuals.append(
            Visual(
                name=visual_name,
                geometry=geometry,
                material=material,
                origin=geom_origin,
            )
        )
    return visuals


def _joint_limits(model, jnt_id: int, act_id, effort_limit: float):
    """! Position and effort limits for a hinge/slide joint.

    MJCF allows limits to be specified either at the joint level (range,
    actuatorfrcrange) or at the actuator level (ctrlrange, forcerange) driving
    the joint. The joint-level values take precedence; when they are missing,
    the actuator-level ones are used.

    @param model The compiled mujoco.MjModel.
    @param jnt_id Joint id.
    @param act_id Id of the actuator driving the joint (None if there is none).
    @param effort_limit Default effort limit.
    @return Tuple (lower, upper, effort) for the URDF joint limit.
    """
    jnt_limited = bool(model.jnt_limited[jnt_id])
    jnt_range = np.asarray(model.jnt_range[jnt_id], dtype=float)
    act_ctrl_limited = act_id is not None and bool(model.actuator_ctrllimited[act_id])
    if jnt_limited:
        lo, hi = float(jnt_range[0]), float(jnt_range[1])
    elif act_ctrl_limited:
        # position limit given at the actuator level (ctrlrange)
        ctrlrange = np.asarray(model.actuator_ctrlrange[act_id], dtype=float)
        lo, hi = float(ctrlrange[0]), float(ctrlrange[1])
    else:
        lo, hi = -1e9, 1e9  # unlimited
    if bool(model.jnt_actfrclimited[jnt_id]):
        # actuator force range given at the joint level
        effort = float(np.max(np.abs(model.jnt_actfrcrange[jnt_id])))
    elif act_id is not None and bool(model.actuator_forcelimited[act_id]):
        # force limit given at the actuator level (forcerange)
        effort = float(np.max(np.abs(model.actuator_forcerange[act_id])))
    else:
        effort = effort_limit
    return lo, hi, effort


def _convert(
    spec,
    model,
    mesh_files: dict,
    mesh_scales: dict,
    effort_limit: float,
    velocity_limit: float,
) -> URDF:
    """! Convert a compiled MuJoCo model into a URDF object."""
    import mujoco

    # Model (robot) name
    name = spec.modelname
    if not name:
        try:
            name = model.names.decode().split("\x00")[0]
        except Exception:
            name = "mujoco_model"
    if not name:
        name = "mujoco_model"

    urdf = URDF(name=name)

    # Map each joint to the actuator driving it, so that actuator-level limits
    # (ctrlrange/forcerange) can be used when the joint itself has no limits
    jnt_actuator = {}
    for act_id in range(model.nu):
        if int(model.actuator_trntype[act_id]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            jnt_actuator.setdefault(int(model.actuator_trnid[act_id, 0]), act_id)

    # One pass over the bodies (body 0 is the worldbody -> root link). Links
    # and joints are added in body order, with the dummy "{joint}_jointbody"
    # links right after their body link.
    for body_id in range(model.nbody):
        child_name = _body_name(model, mujoco, body_id)

        if body_id == 0:
            inertial = Inertial(
                mass=0.0,
                inertia=Inertia(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                origin=Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]),
            )
        else:
            inertial = _make_inertial(
                model.body_mass[body_id],
                model.body_ipos[body_id],
                model.body_inertia[body_id],
                model.body_iquat[body_id],
            )
        visuals = _make_visuals(model, mujoco, body_id, mesh_files, mesh_scales)
        link = Link(name=child_name, inertial=inertial)
        for visual in visuals:
            link.visuals.append(visual)
        urdf.add_link(link)

        if body_id == 0:
            continue  # the worldbody is the root, it has no parent joint

        # Create joints: bodies without joints get a fixed joint to their
        # parent, bodies with joints get a chain of dummy links.
        parent_id = model.body_parentid[body_id]
        parent_name = _body_name(model, mujoco, parent_id)

        body_pos = np.asarray(model.body_pos[body_id], dtype=float)
        body_rpy = _quat2rpy(model.body_quat[body_id])
        body_R = _quat2r(model.body_quat[body_id])

        jnt_num = model.body_jntadr[body_id]
        jnt_count = model.body_jntnum[body_id]

        if jnt_count == 0:
            # Fixed joint to parent, origin is the body pose wrt the parent
            joint = Joint(
                name=f"{parent_name}2{child_name}_fixed",
                parent=parent_name,
                child=child_name,
                joint_type="fixed",
                origin=Pose(xyz=list(body_pos), rpy=list(body_rpy)),
            )
            urdf.add_joint(joint)
            continue

        current_parent = parent_name
        prev_jnt_pos = None
        last_jnt_name = None
        last_jnt_pos = np.zeros(3)

        for j in range(jnt_count):
            jnt_id = jnt_num + j
            jnt_name = _joint_name(model, mujoco, jnt_id)
            jnt_type = int(model.jnt_type[jnt_id])
            jnt_pos = np.asarray(model.jnt_pos[jnt_id], dtype=float)  # anchor in body frame
            jnt_axis = np.asarray(model.jnt_axis[jnt_id], dtype=float)  # axis in body frame
            jnt_limited = bool(model.jnt_limited[jnt_id])
            jnt_range = np.asarray(model.jnt_range[jnt_id], dtype=float)

            # Dummy link located at the joint anchor
            dummy_name = f"{jnt_name}_jointbody"
            urdf.add_link(
                Link(
                    name=dummy_name,
                    inertial=Inertial(
                        mass=DUMMY_LINK_MASS,
                        inertia=Inertia(
                            DUMMY_LINK_INERTIA, 0.0, 0.0,
                            DUMMY_LINK_INERTIA, 0.0, DUMMY_LINK_INERTIA,
                        ),
                        origin=Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]),
                    ),
                )
            )

            # Joint origin (frame of the dummy link) wrt the current parent
            if j == 0:
                origin_xyz = body_pos + body_R @ jnt_pos
                origin_rpy = body_rpy
            else:
                origin_xyz = jnt_pos - prev_jnt_pos
                origin_rpy = np.zeros(3)

            if jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                act_id = jnt_actuator.get(jnt_id)
                lo, hi, effort = _joint_limits(model, jnt_id, act_id, effort_limit)
                if jnt_limited or (act_id is not None and bool(model.actuator_ctrllimited[act_id])):
                    joint_type = "revolute"
                else:
                    joint_type = "continuous"
                limit = JointLimit(lower=lo, upper=hi, effort=effort, velocity=velocity_limit)
            elif jnt_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
                act_id = jnt_actuator.get(jnt_id)
                lo, hi, effort = _joint_limits(model, jnt_id, act_id, effort_limit)
                joint_type = "prismatic"
                limit = JointLimit(lower=lo, upper=hi, effort=effort, velocity=velocity_limit)
            elif jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
                # A free joint fully determines the body pose; use a zero
                # origin so that the joint state q is the body pose itself
                # (same convention as URDF floating joints in optas).
                joint_type = "floating"
                limit = None
                origin_xyz = np.zeros(3)
                origin_rpy = np.zeros(3)
            else:
                raise NotImplementedError(
                    f"MuJoCo joint type {jnt_type} (joint '{jnt_name}') is not supported"
                )

            urdf.add_joint(
                Joint(
                    name=jnt_name,
                    parent=current_parent,
                    child=dummy_name,
                    joint_type=joint_type,
                    origin=Pose(xyz=list(origin_xyz), rpy=list(origin_rpy)),
                    axis=list(jnt_axis) if joint_type != "fixed" else None,
                    limit=limit,
                )
            )

            current_parent = dummy_name
            prev_jnt_pos = jnt_pos
            last_jnt_name = jnt_name
            last_jnt_pos = jnt_pos

        # Fixed joint bringing the chain back to the body (link) frame
        urdf.add_joint(
            Joint(
                name=f"{last_jnt_name}_offset",
                parent=current_parent,
                child=child_name,
                joint_type="fixed",
                origin=Pose(xyz=list(-last_jnt_pos), rpy=[0.0, 0.0, 0.0]),
            )
        )

    return urdf


def mjcf_to_urdf(
    xml_filename: str,
    effort_limit: float = DEFAULT_EFFORT_LIMIT,
    velocity_limit: float = DEFAULT_VELOCITY_LIMIT,
) -> URDF:
    """! Extract a robot model from a MuJoCo XML (MJCF) file.

    @param xml_filename Path to the .xml (MJCF) file. Included files are
           resolved relative to this file by the mujoco compiler.
    @param effort_limit Effort limit written to URDF joint limits (MJCF joints
           carry no effort limit).
    @param velocity_limit Velocity limit written to URDF joint limits (MJCF
           joints carry no velocity limit).
    @return A URDF object (urdf_parser_py) describing the model.
    """
    spec, model, mesh_files, mesh_scales = _spec_and_model(xml_filename)
    return _convert(spec, model, mesh_files, mesh_scales, effort_limit, velocity_limit)


def mjcf_string_to_urdf(
    xml_string: str,
    effort_limit: float = DEFAULT_EFFORT_LIMIT,
    velocity_limit: float = DEFAULT_VELOCITY_LIMIT,
) -> URDF:
    """! Extract a robot model from a MuJoCo XML (MJCF) string.

    Note, MJCF files that use <include> or mesh assets generally require a
    filename so that mujoco can resolve them; prefer mjcf_to_urdf in that case.

    @param xml_string MJCF content as a string.
    @param effort_limit Effort limit written to URDF joint limits.
    @param velocity_limit Velocity limit written to URDF joint limits.
    @return A URDF object (urdf_parser_py) describing the model.
    """
    import mujoco  # lazy import, only needed for .xml files

    spec = mujoco.MjSpec.from_string(xml_string)
    model = spec.compile()
    mesh_files = {}
    mesh_scales = {}
    for mesh in spec.meshes:
        if mesh.name is not None and mesh.file is not None:
            mesh_files[mesh.name] = mesh.file
            scale_attr = getattr(mesh, "scale", None)
            scale = np.asarray(scale_attr if scale_attr is not None else [1.0, 1.0, 1.0], dtype=float)
            mesh_scales[mesh.name] = None if np.allclose(scale, 1.0) else [float(s) for s in scale]
    return _convert(spec, model, mesh_files, mesh_scales, effort_limit, velocity_limit)
