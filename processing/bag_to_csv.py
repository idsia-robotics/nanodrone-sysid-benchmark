import functools as ft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pandas as pd
from scipy.signal import correlate, find_peaks

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
print(f"🧭 Working directory set to project root:\n{PROJECT_ROOT}")

from simulator.utils.plots import *
from simulator.utils.quat import quat_to_euler
from utils.topic_utils import *


def bag_to_csv(experiment_name, run):
    h5_dir = os.path.join('data','real', 'raw', 'rosbags')
    bag_name = f'rosbag2_{experiment_name}_run{run}'
    h5_path = os.path.join(h5_dir, f'{bag_name}.h5')

    ### Load HDF5 bag files
    dfs = dict()

    with pd.HDFStore(h5_path, 'r') as store:
        for key in store.keys():
            df = pd.read_hdf(h5_path, key=key)
            dfs[key] = df

    ### Extract data from individual topics

    topic_map = {
        '/poses': ft.partial(extract_pose, prefix=['poses.0.pose']),
        '/cf/status': extract_status,
        '/cf/pose': ft.partial(extract_pose, prefix=['pose']),
        '/cf/control_radio': ft.partial(extract_controls, prefix=['values']),
        '/cf/image_metadata': extract_metadata,
        '/cf/image_odom': extract_odom,
        '/cf/image_accel': ft.partial(extract_lin_accel, prefix=['accel.accel.linear']),
        '/cf/setpoint': extract_odom,
        '/cf/motors': ft.partial(extract_motors),
    }

    extract_dfs = {}
    for topic, extract_fn in topic_map.items():
        msg_df = dfs[topic]
        # ROS timestamp, message received by ros2 bag record
        timestamp = msg_df.timestamp / 1e9
        # ROS timestamp, message published by its publisher (prefer if available)
        if 'header.stamp.sec' in msg_df:
            timestamp = msg_df['header.stamp.sec'] + msg_df['header.stamp.nanosec'] / 1e9
        df = extract_fn(msg_df)
        df['t'] = timestamp - timestamp.iloc[0]
        extract_dfs[topic] = df


    #%%
    wifi_topics = {
        '/cf/image_odom': 'state_stm32_timestamp',
        '/cf/image_accel': 'state_stm32_timestamp',
        '/cf/motors': 'state_stm32_timestamp',
        '/cf/setpoint': 'setpoint_stm32_timestamp',
    }
    metadata_topic = '/cf/image_metadata'

    extract_dfs = retime_wifi_topics(extract_dfs, wifi_topics, metadata_topic)
    clock_delays = estimate_clock_delays(
        extract_dfs=extract_dfs,
        latency_ref_base='/poses',
        latency_ref_fields=['x', 'y', 'z'],  # or ['vx', 'vy', 'vz'], or ['ax', 'ay', 'az']
        latency_ref_topic={
            'mocap': '/poses',
            'radio': '/cf/pose',
            'wifi':  '/cf/image_odom',
        },
        fs=100.0,
        plot=True
    )

    # Example source mapping
    topic_sources = {
        '/poses': 'mocap',
        '/cf/pose': 'radio',
        '/cf/status': 'radio',
        '/cf/control_radio': 'radio',
        '/cf/image_metadata': 'wifi',
        '/cf/image_odom': 'wifi',
        '/cf/image_accel': 'wifi',
        '/cf/motors': 'wifi',
        '/cf/setpoint': 'wifi',
    }

    # Apply clock alignment
    extract_dfs = apply_clock_delays(extract_dfs, topic_sources, clock_delays)

    # Here I compare the recorded data from the different data sources:
    # - State estimation streamed over Wi-Fi - all components (pos, vel, orient, ang. vel) but has occasional packet drops
    # - State estimation streamed over radio - pos only, but more reliable
    # - Motion capture - pos only
    #
    # - Setpoints streamed over Wi-Fi - pos and vel only, the rest are controlled reactively by the onboard PIDs

    state_wifi_df = extract_dfs['/cf/image_odom']
    state_radio_df = extract_dfs['/cf/pose']
    mocap_df = extract_dfs['/poses']
    setpoint_df = extract_dfs['/cf/setpoint']

    # 1️⃣ Detect flying window
    t_min, t_max = get_flight_window(extract_dfs, status_topic='/cf/status')

    # 2️⃣ Crop all topics to that interval
    topics_to_crop = [
        '/poses', '/cf/motors', '/cf/image_odom', '/cf/image_accel',
        '/cf/setpoint', '/cf/status', '/cf/control_radio'
    ]
    extract_dfs = crop_topics_to_flight(extract_dfs, t_min, t_max, topics_to_crop)

    # 3️⃣ Merge aligned topics
    merge_order = [
        '/poses', '/cf/motors', '/cf/control_radio',
        '/cf/image_accel', '/cf/setpoint', '/cf/status'
    ]
    merged_df = merge_topics(extract_dfs, base_topic='/cf/image_odom', merge_order=merge_order)

    t_min = merged_df['t'].min()
    t_max = merged_df['t'].max()

    # Generate uniformly spaced time vector at 100 Hz (i.e. every 0.01s)
    t_uniform = np.arange(t_min, t_max, 0.01)

    resampled_df = merged_df.set_index('t').reindex(t_uniform, method="nearest").reset_index()
    resampled_df.rename(columns={'index': 't'}, inplace=True)

    resampled_df.to_csv(os.path.join('data/real/raw/csv', experiment_name + f"_run{run}" + '.csv'))


def main():
    for run in [1, 2, 3, 4]:
        experiment_name = 'chirp_20251017'
        bag_to_csv(experiment_name, run)

if __name__ == "__main__":
    main()