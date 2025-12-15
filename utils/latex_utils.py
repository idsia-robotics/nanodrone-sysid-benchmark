import numpy as np
import pandas as pd

# ============================================================
# === Format for standard MAE columns
# ============================================================
def bold_best(col):
    arr = np.asarray(col, dtype=float)
    best = np.min(arr)
    out = []
    for x in arr:
        if abs(x - best) < 1e-12:
            out.append(f"\\textbf{{{x:.4f}}}")
        else:
            out.append(f"{x:.4f}")
    return out

# ============================================================
# === Format for 1:50 italic CUMULATIVE SIM-ERROR columns
# ============================================================
def italic_bold_best(col):
    arr = np.asarray(col, dtype=float)
    best = np.min(arr)
    out = []
    for x in arr:
        s = f"{x:.4f}"
        if abs(x - best) < 1e-12:
            out.append(f"\\textbf{{\\textit{{{s}}}}}")
        else:
            out.append(f"\\textit{{{s}}}")
    return out


def print_latex_table_results(rows, H_TARGETS):
    # ============================================================
    # === DataFrame structure (with the italic 1:50 columns)
    # ============================================================
    columns = (
            ["Model"] +
            [f"$p_{{h={h}}}$" for h in H_TARGETS] + [r"$\textit{p_{h=1{:}50}}$"] +
            [f"$v_{{h={h}}}$" for h in H_TARGETS] + [r"$\textit{v_{h=1{:}50}}$"] +
            [f"$R_{{h={h}}}$" for h in H_TARGETS] + [r"$\textit{R_{h=1{:}50}}$"] +
            [f"$\omega_{{h={h}}}$" for h in H_TARGETS] + [r"$\textit{\omega_{h=1{:}50}}$"]
    )

    df = pd.DataFrame(rows, columns=columns)

    # ============================================================
    # === Apply formatting: MAE columns normal-bold, SimErr italic+bold
    # ============================================================
    mae_cols = []
    sim_cols = []

    for col in columns[1:]:
        if "1:50" in col:
            sim_cols.append(col)
        else:
            mae_cols.append(col)

    for col in mae_cols:
        df[col] = bold_best(df[col].astype(float))

    for col in sim_cols:
        df[col] = italic_bold_best(df[col].astype(float))

    # ============================================================
    # === Build LaTeX
    # ============================================================
    latex_body = df.to_latex(
        index=False,
        escape=False,
        header=False,
    ).strip()

    lines = latex_body.splitlines()
    data_rows = lines[1:-1]

    latex_final = r"""
    \begin{table*}[t]
        \centering
        \caption{Numerical performance at $h=1,10,50$. The italic column reports the cumulative simulation error (sum of MAEs over $h=1..50$).}
        \setlength{\tabcolsep}{3pt}
        \scriptsize
        \renewcommand{\arraystretch}{1.2}
        \begin{tabular}{l|cccc|cccc|cccc|cccc}
            \toprule
            & \multicolumn{4}{c|}{$\mathrm{MAE}_{p,h}$ [m]}
            & \multicolumn{4}{c|}{$\mathrm{MAE}_{v,h}$ [m/s]}
            & \multicolumn{4}{c|}{$\mathrm{MAE}_{R,h}$ [rad]}
            & \multicolumn{4}{c}{$\mathrm{MAE}_{\omega,h}$ [rad/s]}\\[1mm]

            Model
            & $h{=}1$ & $h{=}10$ & $h{=}50$ & \textit{$h{=}1{:}50$}
            & $h{=}1$ & $h{=}10$ & $h{=}50$ & \textit{$h{=}1{:}50$}
            & $h{=}1$ & $h{=}10$ & $h{=}50$ & \textit{$h{=}1{:}50$}
            & $h{=}1$ & $h{=}10$ & $h{=}50$ & \textit{$h{=}1{:}50$}\\
            \midrule
    """ + "\n".join(data_rows) + r"""
            \bottomrule
        \end{tabular}
        \label{tab:numerical_performance}
    \end{table*}
    """

    print(latex_final)