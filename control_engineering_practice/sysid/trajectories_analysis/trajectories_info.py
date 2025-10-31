import os
import numpy as np
import pandas as pd


def describe_trajectory(df, is_real=False):
    """
    Compute summary statistics for a trajectory DataFrame.

    If is_real = True:
        use columns x, vx, ax, pitch, roll, ...
    Else:
        use reference columns x_r, vx_r, pitch_r, roll_r, ...
    """

    suffix = "" if is_real else "_r"

    # --- Timing info ---
    t = df["t"].to_numpy()
    duration = t[-1] - t[0]
    samples = len(t)
    fs = 100

    # --- Positions, velocities ---
    pos = df[[f"x{suffix}", f"y{suffix}", f"z{suffix}"]].to_numpy()
    vel = df[[f"vx{suffix}", f"vy{suffix}", f"vz{suffix}"]].to_numpy()
    speed = np.linalg.norm(vel, axis=1)
    max_speed = np.nanmax(speed)

    # --- Acceleration ---
    if is_real and all(c in df.columns for c in ["ax", "ay", "az"]):
        acc = df[["ax", "ay", "az"]].to_numpy()

        # Subtract gravity in Gs (accelerometers measure total accel including gravity)
        acc[:,2] = acc[:,2] - 9.81

        # Convert from Gs to m/s² if in G
        # if np.nanmax(np.abs(acc)) < 9:
        #     acc *= 9.81
    else:
        # Numerical diff from velocities
        acc = np.gradient(vel, np.mean(np.diff(t)), axis=0)

    accel_mag = np.linalg.norm(acc, axis=1)
    accel_mag = accel_mag[~np.isnan(accel_mag)]
    accel_mag = accel_mag[accel_mag < 50]  # remove spikes
    max_accel = np.nanmax(accel_mag)

    # --- Tilt (use pitch, roll directly) ---
    pitch = df[f"pitch{suffix}"].to_numpy()
    roll = df[f"roll{suffix}"].to_numpy()

    # Detect radians → convert to degrees if needed
    if np.nanmean(np.abs(pitch)) < 0.2 and np.nanmean(np.abs(roll)) < 0.2:
        pitch_deg = np.degrees(pitch)
        roll_deg = np.degrees(roll)
    else:
        pitch_deg = pitch
        roll_deg = roll

    # Compute true tilt angle (deviation from vertical)
    tilt = np.degrees(np.arccos(np.cos(np.radians(pitch_deg)) * np.cos(np.radians(roll_deg))))
    max_tilt = np.nanmax(tilt)

    # --- Spatial extent ---
    mins = np.nanmin(pos, axis=0)
    maxs = np.nanmax(pos, axis=0)
    extent = " × ".join(f"{(maxs[i] - mins[i]):.1f}" for i in range(3))

    return {
        "Duration [s]": duration,
        "Samples": samples,
        "Freq [Hz]": fs,
        "Extent [m]": extent,
        "$v_{\mathrm{max}}$ [m/s]": max_speed,
        "$a_{\mathrm{max}}$ [m/s²]": max_accel,
        "Max tilt [°]": max_tilt,
    }

def summarize_runs(real_dfs):
    """Compute mean ± std across multiple real runs."""
    metrics = [describe_trajectory(df, is_real=True) for df in real_dfs]
    df_metrics = pd.DataFrame(metrics)

    summary = {}
    for c in ["$v_{\mathrm{max}}$ [m/s]", "$a_{\mathrm{max}}$ [m/s²]", "Max tilt [°]"]:
        mean, std = df_metrics[c].mean(), df_metrics[c].std()
        summary[c] = f"{mean:.2f} ± {std:.2f}"
    return summary


def main():
    base_sim = "../../data/sim/new"
    base_real = "../../data/real/processed/new"
    names = ["square", "melon", "random1", "random2", "multisine"]

    results = []

    for name in names:
        sim_file = f"{name}_20251017.parquet"
        sim_path = os.path.join(base_sim, sim_file)
        if not os.path.exists(sim_path):
            print(f"⚠️ Missing sim file: {sim_path}")
            continue

        df_sim = pd.read_parquet(sim_path)
        ref_metrics = describe_trajectory(df_sim, is_real=False)

        # --- Load all available real runs ---
        real_dfs = []
        for run in range(1, 6):
            real_file = f"{name}_20251017_run{run}.parquet"
            real_path = os.path.join(base_real, real_file)
            if os.path.exists(real_path):
                df_real = pd.read_parquet(real_path)
                df_real = df_real.rename(columns={'torch_yaw': 'torque_yaw'})
                real_dfs.append(df_real)

        if real_dfs:
            real_summary = summarize_runs(real_dfs)
        else:
            real_summary = {k: "-" for k in ["Max speed [m/s]", "Max accel [m/s²]", "Max tilt [°]"]}

        results.append({
            "Name": name.capitalize(),
            **{k: f"{v:.2f}" if isinstance(v, float) else v for k, v in ref_metrics.items()},
            **{f"{k} (real)": v for k, v in real_summary.items()},
        })

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    # Optionally export to LaTeX
    summary_df.to_latex("trajectory_summary.tex", index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
