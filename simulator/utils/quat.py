import jax
import jax.numpy as jnp

def quat_normalize(q):
    return q / jnp.linalg.norm(q)

def quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def quat_inv(q):
    return quat_conj(q) / jnp.dot(q, q)

def quat_multiply(q1, q2):
    """Quaternion product q = q1 ⊗ q2"""
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
    return jnp.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q"""
    q_conj = jnp.array([-q[0], -q[1], -q[2], q[3]])
    v_quat = jnp.concatenate([v, jnp.array([0.0])])
    return quat_multiply(quat_multiply(q, v_quat), q_conj)[:3]

def quat_derivative(q, omega):
    """Quaternion time derivative from body angular velocity"""
    omega_quat = jnp.array([omega[0], omega[1], omega[2], 0.0])
    dq = 0.5 * quat_multiply(q, omega_quat)
    return dq

def quat_from_two_vectors(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Create a quaternion that rotates vector a to vector b.
    Equivalent to Eigen::Quaternion::FromTwoVectors(a, b)
    Args:
        a: Source vector (3,)
        b: Target vector (3,)
    Returns:
        Quaternion as jnp.ndarray([x, y, z, w])
    """
    eps = 1e-6
    
    a = a / jnp.linalg.norm(a)
    b = b / jnp.linalg.norm(b)
    c = jnp.dot(a, b)

    def normal():
        v = jnp.cross(a, b)
        s = jnp.sqrt((1.0 + c) * 2.0)
        return jnp.concatenate([v / s, jnp.array([0.5 * s])])

    def near_pi():
        nonlocal c
        c = jnp.maximum(-1, c)
        
        M = jnp.stack([a, b])
        _, _, Vt = jnp.linalg.svd(M)
        axis = Vt[2, :]
        
        w2 = (1.0 + c) / 2.0
        real = jnp.sqrt(w2)
        imag = axis * jnp.sqrt(1 - w2)
        
        return jnp.concatenate([imag, real[None]])

    return jax.lax.cond(c > -1 + eps, normal, near_pi)

### Rotation matrix

def quat_to_rotmat(q):
    """
    Convert a unit quaternion [x, y, z, w] to a 3×3 rotation matrix.

    Quaternion Convention:
        - Input: q = [qx, qy, qz, qw]  (scalar-last format)
        - Rotation is active (applies to vectors), right-handed
        - Follows the Hamilton convention: i² = j² = k² = ijk = -1

    Output:
        - Returns rotation matrix R ∈ SO(3)
        - R maps body-frame vectors to world-frame vectors:
              v_world = R @ v_body
        - R also satisfies:
              R = quat_to_rotmat(q)  ⇒  R @ x_body = x_world
              R.T @ x_world = x_body

    Args:
        q: jnp.array([qx, qy, qz, qw]) — scalar-last quaternion

    Returns:
        R: jnp.array shape (3, 3) — rotation matrix body → world
    """
    qx, qy, qz, qw = q

    return jnp.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ])

def rotmat_to_quat(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]
    """
    trace = jnp.trace(R)

    def case_w():
        S = jnp.sqrt(1.0 + trace) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
        return jnp.array([w, x, y, z])

    def case_x():
        S = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
        return jnp.array([w, x, y, z])

    def case_y():
        S = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
        return jnp.array([w, x, y, z])

    def case_z():
        S = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
        return jnp.array([w, x, y, z])

    def pick_case():
        i = jnp.argmax(jnp.array([R[0, 0], R[1, 1], R[2, 2]]))
        return jax.lax.switch(i, [case_x, case_y, case_z])

    return jax.lax.cond(trace > 0.0, case_w, pick_case)

### Euler angles

def euler_to_quat(yaw, pitch, roll):
    """
    Convert ZYX Euler angles (yaw, pitch, roll) to quaternion [x, y, z, w].

    Args:
        yaw: rotation around Z axis [rad]
        pitch: rotation around Y axis [rad]
        roll: rotation around X axis [rad]

    Returns:
        quat: jnp.array([x, y, z, w]) unit quaternion
    """
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quat = jnp.array([qx, qy, qz, qw])
    return quat / jnp.linalg.norm(quat)

def quat_to_euler(q):
    """
    Convert a quaternion [x, y, z, w] to ZYX Euler angles (yaw, pitch, roll).
    
    Args:
        q: jnp.array([x, y, z, w]) quaternion (scalar-last)
        
    Returns:
        angles: jnp.array([yaw, pitch, roll]) Euler angles [rad]
    """
    x, y, z, w = q

    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = jnp.where(jnp.abs(sinp) >= 1,
                      jnp.sign(sinp) * jnp.pi / 2,  # clamp at ±90°
                      jnp.arcsin(sinp))

    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)
    
    return jnp.stack([yaw, pitch, roll])

### Lie algebra

def vec_to_skew(v: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a 3D vector to a 3×3 skew-symmetric matrix in so(3).

    This operation maps a vector ω ∈ ℝ³ to its equivalent matrix 
    representation in the Lie algebra so(3) using the "hat" operator (∧). 

    For v = [x, y, z]^T, this returns the matrix:
        R = [  0  -z   y
               z   0  -x
              -y   x   0 ]

    Args:
        v (jnp.ndarray): A 3D vector.

    Returns:
        jnp.ndarray: A 3×3 skew-symmetric matrix in so(3).
    """
    return jnp.array([
        [  0.0,  -v[2],   v[1]],
        [ v[2],    0.0,  -v[0]],
        [-v[1],   v[0],    0.0]
    ])

def skew_to_vec(R: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a 3×3 skew-symmetric matrix to a 3D vector.

    This is the inverse of the `vec_to_skew` operation, and it extracts the unique
    components from the so(3) Lie algebra element using the "vee" operator (∨). 

    Given a matrix R ∈ so(3), such that:
        R = [  0   -z   y
               z    0  -x
              -y    x   0 ],
    this function returns the vector [x, y, z]^T ∈ ℝ³.

    Args:
        R (jnp.ndarray): A 3×3 skew-symmetric matrix.

    Returns:
        jnp.ndarray: A 3D vector corresponding to the Lie algebra element.
    """
    return jnp.array([R[2, 1], R[0, 2], R[1, 0]])

def quat_SO3_log(q):
    """
    Compute the Lie logarithm map from SO(3) to so(3) for a unit quaternion.

    This function maps a unit quaternion `q ∈ ℍ₁ ⊂ ℝ⁴`, representing a rotation
    in SO(3), to a 3D vector `ω ∈ ℝ³` in the Lie algebra so(3), corresponding
    to the axis-angle representation of the rotation.

    The map is defined piecewise to maintain numerical stability in degenerate
    or near-identity cases. Specifically, it computes:

        log_SO3(q) = θ * a    where    q = [sin(θ/2)*a, cos(θ/2)], ||a|| = 1

    Args:
        q (jnp.ndarray): Unit quaternion of shape (4,) in the format [x, y, z, w]
                         where the scalar part is the last element.

    Returns:
        jnp.ndarray: A 3D vector ω ∈ ℝ³ corresponding to the logarithmic map from SO(3) to so(3).

    Notes:
        - The returned vector is the minimal representation of the rotation, also known
          as the axis-angle vector.
        - The output lives in the tangent space at the identity of SO(3), i.e., so(3).

    Reference:
        The implementation follows the branching structure used in PyPose and jaxlie,
        with some tweaking to ensure that the Jacobian is correct in all corner cases in addition
        to the function itself
    """
    
    v = q[:3]
    w = q[3]
    eps = 1e-6

    norm_sq = jnp.sum(jnp.square(v), axis=-1)
    norm_small = norm_sq < eps**2

    w_abs = jnp.abs(w)
    w_small = w_abs < eps
    
    def case1():  # general case
        norm = jnp.sqrt(norm_sq)
        return 2 * jnp.arctan(norm / w) / norm * v

    def case2():  # w near 0
        norm = jnp.sqrt(norm_sq)
        sign = jnp.where(jnp.signbit(w), -1, +1)
        return (sign * jnp.pi - 2 * w) / norm * v

    def case3():  # near identity
        return (2.0 / w - norm_sq / (3.0 * w**3)) * v

    return jax.lax.cond(
        norm_small,
        case3,
        lambda: jax.lax.cond(w_small, case2, case1),
    )

def quat_so3_exp(omega):
    """
    Exponential map from so(3) to unit quaternion ℍ₁ ⊂ ℝ⁴.

    Maps a rotation vector `ω ∈ ℝ³` (axis-angle) to a unit quaternion `q ∈ ℍ₁ ⊂ ℝ⁴`,
    representing a rotation in SO(3).

    Given a rotation vector ω = θ * a (with ||a|| = 1), this map returns:
        q = [sin(θ/2)*a, cos(θ/2)] ∈ ℍ₁

    The function uses a stable Taylor expansion for small angles to avoid numerical
    instability in sin(θ/2)/θ and cos(θ/2).

    Args:
        ω (jnp.ndarray): A 3D vector (axis-angle), shape (3)

    Returns:
        jnp.ndarray: A 4D unit quaternion, shape (4), in the format [x, y, z, w]

    Reference:
        Adapted from PyPose's `so3_Exp` implementation.
    """
    eps = 1e-6

    norm_sq = jnp.sum(jnp.square(omega), axis=-1)
    norm_small = norm_sq < eps**2

    def large():
        theta = jnp.sqrt(norm_sq)
        half_theta = theta / 2
        imag = jnp.sin(half_theta) / theta
        real = _cos_half(theta)
        return jnp.concatenate([imag * omega, real[None]], axis=-1)

    def small():
        theta2 = norm_sq
        theta4 = theta2 ** 2
        imag = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4
        real = 1.0 - (1.0 /  8.0) * theta2 + (1.0 /  384.0) * theta4
        return jnp.concatenate([imag * omega, real[None]], axis=-1)

    return jax.lax.cond(norm_small, small, large)

def _cos_half(x):
    delta = jnp.pi - x
    is_near_pi = jnp.abs(delta) < 1e-4
    
    def normal():
        return jnp.cos(x/2)

    def near_pi():
        delta2 = delta * delta
        delta3 = delta2 * delta
        return (1/2) * delta - (1 / 48) * delta3
        
    return jax.lax.cond(is_near_pi, near_pi, normal)