import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pandas as pd
import seaborn as sns

from .quat import quat_rotate, quat_to_rotmat

def plot_state_grid(df, des_df=None, orient='euler'):
    """
    Plot a grid of key state variables over time using seaborn.
    """
    state_groups = {
        "Position [m]": ["x", "y", "z"],
        "Velocity [m/s]": ["vx", "vy", "vz"],
    }

    if orient == 'quat':
        state_groups["Orientation (Quaternion)"] = ["qx", "qy", "qz", "qw"]
    elif orient == 'euler':
        state_groups["Orientation (Euler)"] = ["yaw", "pitch", "roll"]
    else:
        raise ValueError('Unknown orientation value')

    state_groups["Angular Velocity [rad/s]"] = ["wx", "wy", "wz"]

    fig, axes = plt.subplots(len(state_groups), 1, figsize=(10, 12), sharex=True)

    for ax, (title, cols) in zip(axes, state_groups.items()):
        palette     = sns.color_palette(n_colors=len(cols))
        des_palette = palette #sns.color_palette('dark', n_colors=len(cols))
        
        for col, color, des_color in zip(cols, palette, des_palette):
            sns.lineplot(
                ax=ax,
                data=df,
                
                x="t", y=col,
                label=col,
                
                color=color,
            )

            if des_df is not None:
                sns.lineplot(
                    ax=ax, 
                    data=des_df, 
                    
                    x="t", y=col,

                    color=des_color,
                    ls='--', lw=0.75,
                )
            
        
        ax.set_ylabel(title)
        ax.legend(loc="best")
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    return fig

def plot_control_grid(df):
    """
    Plot control inputs over time using seaborn.
    """
    control_cols = ["T", "tau_phi", "tau_theta", "tau_psi"]
    Lb = np.array([0.0, -1.0, -1.0, -1.0])
    Ub = np.array([1.0, 1.0, 1.0, 1.0])

    
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in control_cols:
        sns.lineplot(data=df, x="t", y=col, label=col, ax=ax)

    ax.set_ylim(1.1 * np.min(Lb), 1.1 * np.max(Ub))
    ax.set_ylabel("Normalized Control")
    ax.set_xlabel("Time [s]")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    return fig

def points_to_data_length_3d(ax, pt_size, ref_point=(0, 0, 0), axis='x'):
    """
    Convert a point size (e.g. 10pt) to data units in a 3D plot.
    
    Parameters:
        ax: The 3D axis.
        pt_size: Size in points (1 point = 1/72 inch).
        ref_point: 3D point in data coordinates to evaluate the scaling.
        axis: Direction of unit vector ('x', 'y', or 'z') for data-space reference.
        
    Returns:
        data_length: Length in data units corresponding to `pt_size`.
    """
    fig = ax.get_figure()
    dpi = fig.dpi
    pixel_size = pt_size * dpi / 72  # pt → pixels

    # Reference vector in data space
    ref_point = np.asarray(ref_point)
    direction = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis]
    ref_dir = ref_point + np.array(direction)

    # Project both points to 2D display coordinates
    x1, y1, _ = proj3d.proj_transform(*ref_point, ax.get_proj())
    x2, y2, _ = proj3d.proj_transform(*ref_dir, ax.get_proj())

    # Convert display units to pixels
    trans = ax.transData.transform
    dx1, dy1 = trans((x1, y1))
    dx2, dy2 = trans((x2, y2))

    # Pixel distance of 1 data unit in this direction
    pixel_dist = np.hypot(dx2 - dx1, dy2 - dy1)

    # Data units per pixel
    data_per_pixel = 1 / pixel_dist

    # Scale factor
    return pixel_size * data_per_pixel

def plot_3d_frame(f, ax, name=None, s=1, c=None, artist=None):
    """ Draw 3d frame defined by f on axis ax (if provided) or on a new axis otherwise """
    if c is not None:
        if isinstance(c, str):
            r = f'{c}--'
            g = b = f'{c}-'
        else:
            r, g, b = c
        legend = name
        label = None
    else:
        r, g, b = 'r-', 'g-', 'b-'
        legend = None
        label = name

    s = points_to_data_length_3d(ax, pt_size=10, ref_point=(0, 0, 0), axis='x')
    
    xhat = f @ np.array([[0,0,0,1], [s,0,0,1]]).T
    yhat = f @ np.array([[0,0,0,1], [0,s,0,1]]).T
    zhat = f @ np.array([[0,0,0,1], [0,0,s,1]]).T

    if artist is None:
        x = ax.plot(xhat[0,:], xhat[1,:], xhat[2,:], r, alpha=0.5, label=legend) # transformed x unit vector
        y = ax.plot(yhat[0,:], yhat[1,:], yhat[2,:], g, alpha=0.5) # transformed y unit vector
        z = ax.plot(zhat[0,:], zhat[1,:], zhat[2,:], b, alpha=0.5) # transformed z unit vector
        artist = (x[0], y[0], z[0])
    else:
        x, y, z = artist
        x.set_data(xhat[0,:], xhat[1,:]); x.set_3d_properties(xhat[2,:])
        y.set_data(yhat[0,:], yhat[1,:]); y.set_3d_properties(yhat[2,:])
        z.set_data(zhat[0,:], zhat[1,:]); z.set_3d_properties(zhat[2,:])

    return artist

def set_3daxes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_3d_trajectory(df, des_df=None, step=50, arrow_length=0.05):
    """
    Plot 3D trajectory with discrete orientation arrows (z-axis of body frame).
    
    Parameters:
    - df: DataFrame with trajectory data
    - step: sampling step for orientation arrows
    - arrow_length: length of orientation arrows
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    l = ax.plot(df['x'], df['y'], df['z'], label='Trajectory', linewidth=1.5)
    
    if des_df is not None:
        ax.plot(des_df['x'], des_df['y'], des_df['z'], label='Desired', linewidth=0.75, color=l[0].get_color(), ls='--')
    
    ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], c='green', label='Start')
    ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], c='red', label='End')

    # Plot orientation arrows
    trajectory = df[['x', 'y', 'z']].values
    quats = df[['qx', 'qy', 'qz', 'qw']].values

    for i in range(0, len(df), step):
        frame = np.eye(4)
        frame[:3, :3] = quat_to_rotmat(quats[i])
        frame[:3,  3] = trajectory[i]
        plot_3d_frame(frame, ax)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Flight Trajectory')
    ax.legend()
    ax.grid(True)

    set_3daxes_equal(ax)
    
    return fig

def animate_trajectory(df, des_df=None, subsample=1, arrow_length=0.05):
    """
    Create a 3D animation of the trajectory with body orientation arrows.
    
    Parameters:
    - df: DataFrame with state trajectory
    - arrow_length: Length of orientation arrows
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], lw=1.5, label='Trajectory')

    if des_df is not None:
        ax.plot(des_df['x'], des_df['y'], des_df['z'], label='Desired', linewidth=0.75, color=line.get_color(), ls='--')
    
    ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], c='green', label='Start')
    ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], c='red', label='End')
    # point = ax.scatter([], [], [], c='red', label='Quadrotor')
    frame_artist = None
    # xarrow = ax.quiver(0, 0, 0, 0, 0, 0, length=arrow_length, color='r', normalize=True)
    # zarrow = ax.quiver(0, 0, 0, 0, 0, 0, length=arrow_length, color='b', normalize=True)

    ax.set_xlim(df['x'].min() - 0.1, df['x'].max() + 0.1)
    ax.set_ylim(df['y'].min() - 0.1, df['y'].max() + 0.1)
    ax.set_zlim(df['z'].min() - 0.1, df['z'].max() + 0.1)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    title = ax.set_title('t = 0.0s')

    set_3daxes_equal(ax)
    
    trajectory = df[['x', 'y', 'z']].values
    quats = df[['qx', 'qy', 'qz', 'qw']].values
    time = df['t'].values

    def update(frame):
        # nonlocal xarrow, zarrow, 
        nonlocal ax, title, frame_artist

        line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
        line.set_3d_properties(trajectory[:frame+1, 2])
        # point._offsets3d = ([trajectory[frame, 0]],
        #                     [trajectory[frame, 1]],
        #                     [trajectory[frame, 2]])

        tform = np.eye(4)
        tform[:3, :3] = quat_to_rotmat(quats[frame])
        tform[:3,  3] = trajectory[frame]
        frame_artist = plot_3d_frame(tform, ax, artist=frame_artist)
        
        # # Orientation arrow (body z-axis in world frame)
        # origin = trajectory[frame]
        
        # xaxis = np.array([1.0, 0.0, 0.0])
        # xaxis = quat_rotate(quats[frame], xaxis)
        # xarrow.remove()
        # xarrow = ax.quiver(
        #     origin[0], origin[1], origin[2],
        #      xaxis[0],  xaxis[1],  xaxis[2],
        #     length=arrow_length, color='r', normalize=False
        # )
        
        # zaxis = np.array([0.0, 0.0, 1.0])
        # zaxis = quat_rotate(quats[frame], zaxis)
        # zarrow.remove()
        # zarrow = ax.quiver(
        #     origin[0], origin[1], origin[2],
        #      zaxis[0],  zaxis[1],  zaxis[2],
        #     length=arrow_length, color='b', normalize=False
        # )

        title.set_text(f't = {time[frame]:.3f}s')
        
        # return line, point, xarrow, zarrow, title
        return line, *frame_artist, title
    
    dt = df['t'].diff().mean()
    interval = (dt * 1e3) / subsample
    anim = FuncAnimation(fig, update, frames=df.index[::subsample], interval=interval, blit=False)
    return anim
