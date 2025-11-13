import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

from simulator.utils.quat import quat_to_euler

def extract_cols(df, cols, prefix=[]):
    if isinstance(cols, list):
        cols = dict(zip(cols, cols))

    cols = {
        '.'.join(prefix + [old]): new for old, new in cols.items()
    }

    return df[cols.keys()].rename(columns=cols)


def extract_position(df, prefix=[]):
    return extract_cols(df, ['x', 'y', 'z'], prefix)


def extract_orientation(df, prefix=[]):
    df = extract_cols(df, {'x': 'qx', 'y': 'qy', 'z': 'qz', 'w': 'qw'}, prefix)

    quats = df.values  # shape (T, 4)
    euler_angles = jax.vmap(quat_to_euler)(quats)  # shape (T, 3)
    yaw, pitch, roll = euler_angles.T
    df['yaw'] = np.asarray(yaw)
    df['pitch'] = np.asarray(pitch)
    df['roll'] = np.asarray(roll)

    return df


def extract_pose(df, prefix=[]):
    return pd.concat([
        extract_position(df, prefix=prefix + ['position']),
        extract_orientation(df, prefix=prefix + ['orientation']),
    ], axis=1)


def extract_lin_vel(df, prefix=[]):
    return extract_cols(df, {'x': 'vx', 'y': 'vy', 'z': 'vz'}, prefix)


def extract_ang_vel(df, prefix=[]):
    return extract_cols(df, {'x': 'wx', 'y': 'wy', 'z': 'wz'}, prefix)


def extract_twist(df, prefix=[]):
    return pd.concat([
        extract_lin_vel(df, prefix=prefix + ['linear']),
        extract_ang_vel(df, prefix=prefix + ['angular']),
    ], axis=1)


def extract_odom(df, prefix=[]):
    return pd.concat([
        extract_pose(df, prefix=prefix + ['pose.pose']),
        extract_twist(df, prefix=prefix + ['twist.twist']),
    ], axis=1)


def extract_lin_accel(df, prefix=[]):
    return extract_cols(df, {'x': 'ax', 'y': 'ay', 'z': 'az'}, prefix)


def extract_motors(df, prefix=[]):
    return pd.concat([
        extract_cols(df, ["stm32_timestamp"], prefix),
        extract_motor_thrusts(df, prefix=prefix + ['thrust']),
        extract_motor_erpm(df, prefix=prefix + ['erpm']),
    ], axis=1)


def extract_motor_thrusts(df, prefix=[]):
    df = extract_cols(df, {'m1': 'm1_thrust', 'm2': 'm2_thrust', 'm3': 'm3_thrust', 'm4': 'm4_thrust'}, prefix)
    df /= (2 ** 16 - 1)
    return df


def extract_motor_erpm(df, prefix=[]):
    df = extract_cols(df, {'m1': 'm1_erpm', 'm2': 'm2_erpm', 'm3': 'm3_erpm', 'm4': 'm4_erpm'}, prefix)
    df = df.replace(65535, np.nan)
    df *= 100
    return df


def extract_motors(df, prefix=[]):
    return pd.concat([
        extract_motor_thrusts(df, prefix=prefix + ['thrust']),
        extract_motor_erpm(df, prefix=prefix + ['erpm']),
    ], axis=1)


def extract_controls(df, prefix=[]):
    df = extract_cols(df, {'0': 'thrust', '1': 'torque_roll', '2': 'torque_pitch', '3': 'torch_yaw'}, prefix)
    df /= (2 ** 16 - 1)
    return df


def extract_supervisor_info(df, prefix=[]):
    df = extract_cols(df, ['supervisor_info'], prefix)
    df = pd.DataFrame({
        'can_be_armed': df.supervisor_info & (1 << 0),
        'is_armed': df.supervisor_info & (1 << 1),
        'auto_arm': df.supervisor_info & (1 << 2),
        'can_fly': df.supervisor_info & (1 << 3),
        'is_flying': df.supervisor_info & (1 << 4),
        'is_tumbled': df.supervisor_info & (1 << 5),
        'is_locked': df.supervisor_info & (1 << 6),
    }).astype(bool)
    return df


def extract_status(df, prefix=[]):
    return pd.concat([
        extract_cols(df, {'battery_voltage': 'Vbat'}, prefix),
        extract_supervisor_info(df, prefix)
    ], axis=1)


def extract_metadata(df, prefix=[]):
    return extract_cols(df, ['setpoint_priority', 'state_stm32_timestamp', 'setpoint_stm32_timestamp'], prefix)

# # Time synchronization
def retime_wifi_topics(extract_dfs: dict, wifi_topics: dict, metadata_topic: str) -> dict:
    """
    Retime Wi-Fi topics using hardware timestamps from image_metadata.
    ROS timestamps are unreliable due to variable network latency.

    Parameters
    ----------
    extract_dfs : dict[str, pd.DataFrame]
        Extracted topic DataFrames, each containing at least a 't' column (ROS time).
    wifi_topics : dict[str, str]
        Mapping from topic -> field name in metadata containing hardware timestamp.
    metadata_topic : str
        Topic key in extract_dfs containing image metadata.

    Returns
    -------
    dict[str, pd.DataFrame]
        Updated extract_dfs with retimed Wi-Fi topics.
    """
    metadata_df = extract_dfs[metadata_topic]

    for topic, timestamp_field in wifi_topics.items():
        data_df = extract_dfs[topic].copy()

        # Filter out zero or invalid hardware timestamps
        valid_meta = metadata_df[metadata_df[timestamp_field] != 0].copy()
        if valid_meta.empty or data_df.empty:
            print(f"[WARN] Skipping {topic}: no valid metadata or empty data.")
            continue

        # Match messages by nearest ROS timestamp
        merged = pd.merge_asof(
            data_df.sort_values("t"),
            valid_meta.sort_values("t"),
            on="t",
            direction="nearest"
        )

        # Replace ROS timestamp with hardware timestamp (µs → ms → s)
        hw_timestamp = merged[timestamp_field] / 1e3
        data_df["ros_timestamp"] = data_df["t"]
        data_df["t"] = hw_timestamp

        # Drop duplicated timestamps (can happen if multiple messages share a hw time)
        data_df = data_df.drop_duplicates(subset="t")

        extract_dfs[topic] = data_df

    return extract_dfs


def estimate_clock_delays(
    extract_dfs: dict,
    latency_ref_base: str,
    latency_ref_fields: list[str],
    latency_ref_topic: dict,
    fs: float = 100.0,
    plot: bool = True
) -> dict:
    """
    Estimate relative clock delays between sources via multi-dimensional cross-correlation
    of position-, velocity-, or acceleration-like signals.

    Parameters
    ----------
    extract_dfs : dict[str, pd.DataFrame]
        Topic DataFrames, each with time column 't' and numeric signal columns.
    latency_ref_base : str
        Reference topic (e.g., mocap ground truth).
    latency_ref_fields : list[str]
        List of signal columns to use (e.g. ['x', 'y', 'z'] or ['vx', 'vy', 'vz']).
    latency_ref_topic : dict[str, str]
        Mapping from source name ('wifi', 'radio', etc.) → topic name.
    fs : float, optional
        Resampling frequency in Hz for uniform interpolation (default: 100).
    plot : bool, optional
        Whether to generate diagnostic plots.

    Returns
    -------
    dict[str, float]
        Estimated clock delay (seconds) for each source.
    """
    ref_df = extract_dfs[latency_ref_base].set_index("t")
    available_fields = [f for f in latency_ref_fields if f in ref_df.columns]
    if not available_fields:
        raise ValueError(f"No valid reference fields found in {latency_ref_base}")
    ref_data = ref_df[available_fields].to_numpy()
    clock_delays = {}

    if plot:
        fig, axs = plt.subplots(len(latency_ref_topic), 1, figsize=(8, 10))

    for i, (source, topic) in enumerate(latency_ref_topic.items()):
        meas_df = extract_dfs[topic].set_index("t")
        fields = [f for f in latency_ref_fields if f in meas_df.columns]
        if not fields:
            print(f"[WARN] Skipping {topic}: missing fields {latency_ref_fields}")
            continue

        meas_data = meas_df[fields].to_numpy()

        # Common uniform time base (union of both)
        t_min = min(ref_df.index[0], meas_df.index[0])
        t_max = max(ref_df.index[-1], meas_df.index[-1])
        t_uniform = np.arange(t_min, t_max, 1 / fs)

        # Interpolate each dimension
        ref_interp = np.vstack([
            np.interp(t_uniform, ref_df.index, ref_df[col],
                      left=ref_df[col].iloc[0], right=ref_df[col].iloc[-1])
            for col in available_fields
        ]).T
        meas_interp = np.vstack([
            np.interp(t_uniform, meas_df.index, meas_df[col],
                      left=meas_df[col].iloc[0], right=meas_df[col].iloc[-1])
            for col in fields
        ]).T

        # Normalize per dimension
        ref_norm = (ref_interp - np.mean(ref_interp, axis=0)) / np.std(ref_interp, axis=0)
        meas_norm = (meas_interp - np.mean(meas_interp, axis=0)) / np.std(meas_interp, axis=0)

        # Combined cross-correlation (sum over dimensions)
        corr_sum = np.zeros(len(ref_norm) * 2 - 1)
        for j in range(ref_norm.shape[1]):
            corr_sum += correlate(meas_norm[:, j], ref_norm[:, j], mode="full")

        lags = np.arange(-len(ref_norm) + 1, len(meas_norm))
        delay_samples = lags[np.argmax(corr_sum)]
        delay_seconds = delay_samples / fs
        clock_delays[source] = delay_seconds

        # --- Plotting ---
        if plot:
            ax = axs[i] if len(latency_ref_topic) > 1 else axs
            # Plot a representative dimension (first one)
            ax.plot(t_uniform, ref_norm[:, 0], label="Reference")
            ax.plot(t_uniform, meas_norm[:, 0], label="Measured", alpha=0.7)
            ax.plot(t_uniform - delay_seconds, meas_norm[:, 0],
                    label="Aligned", alpha=0.8)
            ax.set_title(f"{source}: Δt = {delay_seconds:+.3f} s")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)

            ax2 = ax.twinx()
            ax2.plot(lags / fs, corr_sum, "r", alpha=0.3)
            ax2.set_ylabel("Combined Cross-correlation")

    if plot:
        fig.suptitle("Multi-dimensional Clock Delay Estimation", fontsize=12)
        fig.tight_layout()

    print("\nEstimated clock delays (seconds):")
    for src, delay in clock_delays.items():
        print(f"  {src:<8}: {delay:+.3f} s")

    return clock_delays

def apply_clock_delays(extract_dfs: dict, topic_sources: dict, clock_delays: dict) -> dict:
        """
        Apply estimated clock delays to each topic's timestamps.

        Parameters
        ----------
        extract_dfs : dict[str, pd.DataFrame]
            Topic DataFrames containing a 't' column.
        topic_sources : dict[str, str]
            Mapping from topic name -> source type ('mocap', 'wifi', 'radio', ...).
        clock_delays : dict[str, float]
            Estimated clock delays in seconds for each source.
            Positive = source lags behind mocap.

        Returns
        -------
        dict[str, pd.DataFrame]
            Updated topic DataFrames with corrected timestamps.
        """
        for topic, source in topic_sources.items():
            if topic not in extract_dfs or extract_dfs[topic].empty:
                print(f"[WARN] Skipping {topic}: missing or empty DataFrame.")
                continue

            if source not in clock_delays:
                print(f"[WARN] No clock delay defined for source '{source}' (topic {topic}).")
                continue

            delay = clock_delays[source]
            extract_dfs[topic]['t'] -= delay
            print(f"{topic:20s}: shifted by {delay:+.3f} s ({source})")

        return extract_dfs

def get_flight_window(extract_dfs: dict, status_topic: str = '/cf/status') -> tuple[float, float]:
    """
    Determine the time interval when the drone is flying.

    Parameters
    ----------
    extract_dfs : dict[str, pd.DataFrame]
        Dictionary of topic DataFrames.
    status_topic : str, optional
        Topic containing the 'is_flying' boolean flag.

    Returns
    -------
    (t_min, t_max) : tuple[float, float]
        Start and end timestamps (seconds) of the flight interval.
    """
    status_df = extract_dfs[status_topic]
    if 'is_flying' not in status_df.columns:
        raise ValueError(f"{status_topic} has no 'is_flying' column.")

    flying_mask = status_df['is_flying'].astype(bool)
    flying_times = status_df.loc[flying_mask, 't']

    if flying_times.empty:
        raise ValueError("No flying interval detected.")

    t_min, t_max = flying_times.min(), flying_times.max()
    print(f"Detected flight window: {t_min:.2f} → {t_max:.2f} s")
    return t_min, t_max


def crop_topics_to_flight(extract_dfs: dict, t_min: float, t_max: float, topics: list[str]) -> dict:
    """
    Restrict each DataFrame to the flying interval.

    Parameters
    ----------
    extract_dfs : dict[str, pd.DataFrame]
        Topic DataFrames with a 't' column.
    t_min, t_max : float
        Flight window (seconds).
    topics : list[str]
        Topics to crop.

    Returns
    -------
    dict[str, pd.DataFrame]
        Updated topic DataFrames.
    """
    for topic in topics:
        df = extract_dfs.get(topic)
        if df is None or 't' not in df.columns:
            print(f"[WARN] Skipping {topic}: missing or invalid DataFrame.")
            continue
        cropped = df[(df['t'] >= t_min) & (df['t'] <= t_max)].copy()
        extract_dfs[topic] = cropped.sort_values('t').reset_index(drop=True)
        print(f"{topic}: kept {len(cropped)} samples.")
    return extract_dfs


def merge_topics(
    extract_dfs: dict,
    base_topic: str = '/cf/image_odom',
    merge_order: list[str] = None
) -> pd.DataFrame:
    """
    Merge multiple topic DataFrames on their timestamps using nearest-neighbor alignment.

    Parameters
    ----------
    extract_dfs : dict[str, pd.DataFrame]
        Dictionary of topic DataFrames.
    base_topic : str, optional
        Topic used as reference timeline.
    merge_order : list[str], optional
        Topics to merge sequentially. If None, merges all except the base.

    Returns
    -------
    merged_df : pd.DataFrame
        Combined DataFrame with aligned signals.
    """
    if merge_order is None:
        merge_order = [k for k in extract_dfs.keys() if k != base_topic]

    merged_df = extract_dfs[base_topic].sort_values('t').copy()
    print(f"Starting merge from base topic: {base_topic}")

    for topic in merge_order:
        df = extract_dfs.get(topic)
        if df is None or 't' not in df.columns:
            print(f"[WARN] Skipping {topic}: missing or invalid DataFrame.")
            continue

        suffix = f"_{topic.split('/')[-1]}"  # add topic-based suffix
        merged_df = pd.merge_asof(
            merged_df.sort_values('t'),
            df.sort_values('t'),
            on='t',
            direction='nearest',
            suffixes=('', suffix)
        )
        print(f"  merged: {topic} ({len(df)} samples)")

    # Normalize time to start at zero
    merged_df['t'] -= merged_df['t'].iloc[0]
    # Example axis correction (specific to Crazyflie convention)
    if 'wy' in merged_df.columns:
        merged_df['wy'] = -merged_df['wy']

    print(f"\nMerged DataFrame shape: {merged_df.shape}")
    return merged_df


