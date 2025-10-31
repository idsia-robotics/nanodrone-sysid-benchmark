import numpy as np

def export_traj_for_uav_trajectories(traj, filename="trajectory.csv"):
    """
    Export a sampled trajectory (Setpoint pytree) into the waypoint format
    expected by uav_trajectories.

    Format (CSV with header):
    t,x,y,z,vx,vy,vz,ax,ay,az,omega_x,omega_y,omega_z,yaw

    Args:
        traj: trajectory pytree (with .pos, .vel, .acc, .omega, .orientation, .t)
        filename: output CSV file
        n_points: if given, number of waypoints to keep (downsampled).
                  If None, defaults to full trajectory length.
    """
    pos = np.array(traj.pos)          # (N,3)
    vel = np.array(traj.vel)          # (N,3)
    acc = np.array(traj.accel)          # (N,3)
    omega = np.array(traj.omega)      # (N,3)
    t_vec  = np.array(traj.t)            # (N,)
    quats = np.array(traj.orientation)  # (N,4) [x,y,z,w]
    yaws = np.array(traj.yaw)
    # choose how many points to keep
    n_points = len(t_vec)

    # indices for downsampling
    idx = np.linspace(0, len(t_vec) - 1, n_points, dtype=int)

    with open(filename, "w") as f:
        f.write("t,x,y,z,vx,vy,vz,ax,ay,az,wx,wy,wz,yaw\n")
        for i in range(len(t_vec)):
            x, y, z = pos[i]
            vx, vy, vz = vel[i]
            ax, ay, az = acc[i]
            wx, wy, wz = omega[i]

            # extract yaw from quaternion
            qx, qy, qz, qw = quats[i]
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = yaws[i]#np.arctan2(siny_cosp, cosy_cosp)

            f.write(
                f"{t_vec[i]:.3f},{x:.3f},{y:.3f},{z:.3f},"
                f"{vx:.3f},{vy:.3f},{vz:.3f},"
                f"{ax:.3f},{ay:.3f},{az:.3f},"
                f"{wx:.3f},{wy:.3f},{wz:.3f},{yaw:.3f}\n"
            )

    print(f"✅ Trajectory exported to {filename} with {n_points} waypoints")

# def export_traj_for_uav_trajectories(traj, filename="trajectory.csv", n_points=None):
#     """
#     Export a sampled trajectory (Setpoint pytree) into the waypoint format
#     expected by uav_trajectories.
#
#     Format (CSV with header): t,x,y,z,yaw
#
#     Args:
#         traj: trajectory pytree (with .pos, .t, .orientation)
#         filename: output CSV file
#         n_points: if given, number of waypoints to keep (downsampled).
#                   If None, defaults to ~1 waypoint per second.
#     """
#     pos = np.array(traj.pos)          # (N,3)
#     ts  = np.array(traj.t)            # (N,)
#     quats = np.array(traj.orientation)  # (N,4) [x,y,z,w]
#
#     # choose how many points to keep
#     if n_points is None:
#         duration = ts[-1] - ts[0]
#         n_points = max(5, int(duration))  # at least 5 points
#
#     # indices for downsampling
#     idx = np.linspace(0, len(ts)-1, n_points, dtype=int)
#
#     with open(filename, "w") as f:
#         f.write("t,x,y,z,yaw\n")
#         for i in idx:
#             x, y, z = pos[i]
#
#             # extract yaw from quaternion
#             qx, qy, qz, qw = quats[i]
#             siny_cosp = 2.0 * (qw * qz + qx * qy)
#             cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
#             yaw = np.arctan2(siny_cosp, cosy_cosp)
#
#             f.write(f"{ts[i]:.3f},{x:.3f},{y:.3f},{z:.3f},{yaw:.3f}\n")
#
#     print(f"✅ Trajectory exported to {filename} with {n_points} waypoints")