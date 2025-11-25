import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from pytorch3d.transforms import (
    quaternion_to_axis_angle,
    axis_angle_to_quaternion,
)

def quat_xyzw_to_wxyz(q):
    # (x,y,z,w) → (w,x,y,z)
    return torch.cat([q[..., 3:], q[..., :3]], dim=-1)

def quat_wxyz_to_xyzw(q):
    # (w,x,y,z) → (x,y,z,w)
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)

# ============================================================
# === Base class for all models (handles 1-step / multi-step)
# ============================================================

class BaseQuadModel(nn.Module):
    def __init__(self, dt=0.01):
        super().__init__()
        self.dt = dt

    def one_step(self, x, u):
        """Override in subclasses: (x,u) -> x_next"""
        raise NotImplementedError

    def forward(self, x0, u_seq):
        """
        Handles both one-step (B,1,4) and multi-step (B,N,4) cases.
        Returns trajectory [B,N,state_dim].
        """
        if u_seq.ndim == 2:  # (B,4)
            u_seq = u_seq.unsqueeze(1)
        if x0.ndim == 2:  # (B,state)
            x = x0
        else:
            x = x0.squeeze(1)

        B, N, _ = u_seq.shape
        preds = []
        for t in range(N):
            u_t = u_seq[:, t, :]
            x = self.one_step(x, u_t)
            preds.append(x.unsqueeze(1))
        return torch.cat(preds, dim=1)


# ============================================================
# === 1. Physics model
# ============================================================
class PhysQuadModel(BaseQuadModel):
    """
    State x = [pos(3), vel(3), so3(3), omega(3)]
    one_step expects u_mot: (B,4) in rad/s; motor_to_phys converts to normalized [T,τ].
    """

    def __init__(self, params, dt, arm_length=0.0353, Kt=3.72e-08, Kc=7.74e-12):
        super().__init__(dt)
        self.m = params["m"]
        self.g = params["g"]
        self.thrust_to_weight = params["thrust_to_weight"]

        self.register_buffer("J", torch.as_tensor(params["J"], dtype=torch.float32))
        self.register_buffer("J_inv", torch.linalg.inv(self.J))
        self.register_buffer("max_torque", torch.as_tensor(params["max_torque"], dtype=torch.float32).view(-1))
        self.register_buffer("gravity", torch.tensor([0.0, 0.0, self.g], dtype=torch.float32))

        # Motor model constants
        self.arm = arm_length
        self.Kt = Kt
        self.Kc = Kc
        self.T_max = self.thrust_to_weight * self.m * self.g

    @torch.no_grad()
    def motor_to_phys(self, u_mot):
        ω2 = u_mot ** 2
        T = self.Kt * ω2.sum(dim=1)
        τx = self.Kt * self.arm * ((ω2[:, 2] + ω2[:, 3]) - (ω2[:, 0] + ω2[:, 1]))
        τy = self.Kt * self.arm * ((ω2[:, 1] + ω2[:, 2]) - (ω2[:, 0] + ω2[:, 3]))
        τz = self.Kc * -((ω2[:, 0] + ω2[:, 2]) - (ω2[:, 1] + ω2[:, 3]))
        T_norm = T / self.T_max
        τ_norm = torch.stack([τx, τy, τz], dim=1) / self.max_torque  # (B,3)
        return torch.cat([T_norm.unsqueeze(1), τ_norm], dim=1)  # (B,4)

    def one_step(self, x, u_mot):
        """x: (B,12), u_mot: (B,4) rad/s"""
        u_phys = self.motor_to_phys(u_mot)  # (B,4) normalized physics inputs

        # ---- unpack external state and convert log(SO3) -> quaternion once ----
        pos = x[:, 0:3]  # (B,3)
        vel = x[:, 3:6]  # (B,3)
        so3 = x[:, 6:9]  # (B,3)
        omega = x[:, 9:12]  # (B,3)

        # --- log(SO3) -> quaternion ---
        quat = self.so3_log_to_quat(so3)  # (B,4), kept internally during RK4
        # --- build quaternion state for physics ---
        x = torch.cat([pos, vel, quat, omega], dim=-1)
        # --- physics step in quaternion space (your trusted integrator) ---
        x_next = self._step_from_phys(x, u_phys)

        # --- unpack ---
        pos_next = x_next[:, 0:3]  # (B,3)
        vel_next = x_next[:, 3:6]  # (B,3)
        quat_next = x_next[:, 6:10]  # (B,3)
        omega_next = x_next[:, 10:13]  # (B,3)

        # ---- convert back to log(SO3) only once at the end ----
        so3_next = self.quat_to_so3_log(quat_next)

        # final state in external representation
        x_next = torch.cat([pos_next, vel_next, so3_next, omega_next], dim=-1)
        return x_next

    def _step_from_phys(self, x, u_phys):
        """
        RK4 integration of quadrotor rigid-body dynamics.

        External state: x = [pos(3), vel(3), so3_log(3), omega(3)]
        Internal integration state: [pos(3), vel(3), quat(4), omega(3)]
        """

        dt = self.dt

        # ---- unpack external state and convert log(SO3) -> quaternion once ----
        pos = x[:, 0:3]  # (B,3)
        vel = x[:, 3:6]  # (B,3)
        quat = x[:, 6:10]  # (B,3)
        omega = x[:, 10:13]  # (B,3)

        # ---- dynamics in quaternion space ----
        def f(pos, vel, quat, omega, u):
            """
            Compute time derivatives (pos_dot, vel_dot, quat_dot, omega_dot)
            given current state and controls.

            pos, vel, quat, omega all have shape (B,3)/(B,4).
            u: (B,4) = [T_norm, τ_norm]
            """
            # --- controls ---
            T_norm = torch.clamp(u[:, 0], 0.0, 1.0)
            tau_norm = torch.clamp(u[:, 1:], -1.0, 1.0)
            T = T_norm * (self.thrust_to_weight * self.m * self.g)
            tau = tau_norm * self.max_torque  # (B,3)

            # --- translational dynamics ---
            thrust_b = torch.zeros_like(vel)
            thrust_b[:, 2] = T
            thrust_w = self.quat_rotate(quat, thrust_b)
            acc = (thrust_w - self.m * self.gravity) / self.m

            # --- rotational dynamics ---
            Jω = omega @ self.J.T
            omega_dot = 0.0*torch.linalg.solve(
                self.J, (tau - torch.cross(omega, Jω, dim=-1)).unsqueeze(-1)
            ).squeeze(-1)

            # --- quaternion derivative ---
            quat_dot = self.quat_derivative(quat, omega)

            # pos_dot = vel
            return vel, acc, quat_dot, omega_dot

        # ---- RK4 stages in quaternion space ----
        # k1
        v1, a1, qd1, w1 = f(pos, vel, quat, omega, u_phys)

        # k2
        pos2 = pos + 0.5 * dt * v1
        vel2 = vel + 0.5 * dt * a1
        quat2 = F.normalize(quat + 0.5 * dt * qd1, dim=-1)
        omega2 = omega + 0.5 * dt * w1
        v2, a2, qd2, w2 = f(pos2, vel2, quat2, omega2, u_phys)

        # k3
        pos3 = pos + 0.5 * dt * v2
        vel3 = vel + 0.5 * dt * a2
        quat3 = F.normalize(quat + 0.5 * dt * qd2, dim=-1)
        omega3 = omega + 0.5 * dt * w2
        v3, a3, qd3, w3 = f(pos3, vel3, quat3, omega3, u_phys)

        # k4
        pos4 = pos + dt * v3
        vel4 = vel + dt * a3
        quat4 = F.normalize(quat + dt * qd3, dim=-1)
        omega4 = omega + dt * w3
        v4, a4, qd4, w4 = f(pos4, vel4, quat4, omega4, u_phys)

        # ---- RK4 integrate ----
        pos_next = pos + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        vel_next = vel + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        omega_next = omega + (dt / 6.0) * (w1 + 2 * w2 + 2 * w3 + w4)

        quat_next = quat + (dt / 6.0) * (qd1 + 2 * qd2 + 2 * qd3 + qd4)
        quat_next = F.normalize(quat_next, dim=-1)

        # final state in external representation
        x_next = torch.cat([pos_next, vel_next, quat_next, omega_next], dim=-1)
        return x_next

    # ======================================================
    # === Quaternion utilities ===
    # ======================================================
    @staticmethod
    def quat_to_so3_log(q_xyzw):
        """
        q_xyzw: (...,4) quaternion in (x,y,z,w)
        returns rotation vector r in R^3
        """
        q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
        r = quaternion_to_axis_angle(q_wxyz)  # (...,3)
        return r


    @staticmethod
    def so3_log_to_quat(r):
        """
        r: (...,3) rotation vector
        returns quaternion q_xyzw in (x,y,z,w)
        """
        q_wxyz = axis_angle_to_quaternion(r)  # (...,4)
        q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
        return q_xyzw

    @staticmethod
    def quat_rotate(q, v):
        qv = q[:, :3]
        qw = q[:, 3:4]
        t = 2 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

    @staticmethod
    def quat_derivative(q, omega):
        qv = q[:, :3]
        qw = q[:, 3:4]
        dqv = 0.5 * (qw * omega + torch.cross(qv, omega, dim=-1))
        dqw = -0.5 * (qv * omega).sum(dim=-1, keepdim=True)
        return torch.cat([dqv, dqw], dim=-1)

# ============================================================
# === 3. Neural model
# ============================================================

class NeuralQuadModel(BaseQuadModel):
    def __init__(self, state_dim=12, input_dim=4, hidden_dim=512, num_layers=4, dt=0.01):
        super().__init__(dt)
        layers = []
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        dim = state_dim + input_dim
        for i in range(num_layers):
            layers += [nn.Linear(dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, state_dim)
        nn.init.zeros_(self.out.weight)
        # nn.init.normal_(self.out.weight, mean=0, std=1e-4)
        # nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def one_step(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        dx = self.out(self.mlp(xu))
        return x + dx


# ============================================================
# === 4. Residual model (Physics + NN correction)
# ============================================================

class ResidualQuadModel(BaseQuadModel):
    def __init__(self, phys: PhysQuadModel, neural: NeuralQuadModel,
                 x_scaler, u_scaler, eps: float = 0):
        super().__init__(phys.dt)
        self.phys = phys
        self.neural = neural
        self.eps = eps

        # Cache scaler stats as device tensors (avoid CPU<->GPU + sklearn at runtime)
        def _to_tensors(scaler):
            mean = torch.as_tensor(getattr(scaler, "mean_", None), dtype=torch.float32)
            scale = torch.as_tensor(getattr(scaler, "scale_", None), dtype=torch.float32)
            return mean, scale

        x_mean, x_scale = _to_tensors(x_scaler)
        u_mean, u_scale = _to_tensors(u_scaler)

        # register as buffers so they move with .to(device)
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_scale", x_scale)
        self.register_buffer("u_mean", u_mean)
        self.register_buffer("u_scale", u_scale)

    # ---- safe (de)normalization on device ----
    def x_denorm(self, x_norm):
        return x_norm * (self.x_scale) + self.x_mean

    def x_normed(self, x_real):
        return (x_real - self.x_mean) / (self.x_scale)

    def u_denorm(self, u_norm):
        return u_norm * (self.u_scale) + self.u_mean

    def one_step(self, x_norm, u_norm):
        # 1) Denormalize to real space (no CPU hops)
        x_real = self.x_denorm(x_norm)
        u_mot = self.u_denorm(u_norm) # motors

        # 2) physics next state (real → then back to norm)
        with torch.no_grad():
            x_phys_next_real = self.phys.one_step(x_real, u_mot)  # (B,12)
        x_phys_next_norm = self.x_normed(x_phys_next_real)

        # 3) NN predicts *residual step* Δx_res in normalized space
        #    Use the neural head directly to get dx, not x+dx
        xu = torch.cat([x_norm, u_norm], dim=-1)
        dx_res_norm = self.neural.out(self.neural.mlp(xu))  # (B,12), zero-init -> starts at 0

        # 4) combine on top of physics prediction (NOT on top of x_norm)
        x_next_norm = x_phys_next_norm + dx_res_norm

        # 5) numerical safety (optional)
        if not torch.all(torch.isfinite(x_next_norm)):
            x_next_norm = x_phys_next_norm  # drop residual if it goes NaN/Inf

        return x_next_norm

# ============================================================
# === 5. LSTM model
# ============================================================
# class QuadLSTM(BaseQuadModel):
#     def __init__(self, input_dim_u=4, state_dim_x=12, hidden_dim=64, num_layers=2, dt=0.01):
#         super().__init__(dt)
#         self.input_dim_u = input_dim_u
#         self.state_dim_x = state_dim_x
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.dropout = nn.Dropout(p=0.1)
#         self.lstm = nn.LSTM(input_dim_u, hidden_dim, num_layers, batch_first=True)
#         self.h0 = nn.Sequential(
#             nn.Linear(state_dim_x, hidden_dim),
#             nn.Tanh()
#         )
#         self.c0 = nn.Sequential(
#             nn.Linear(state_dim_x, hidden_dim),
#             nn.Tanh()
#         )
#         self.out = nn.Linear(hidden_dim, state_dim_x)
#         nn.init.zeros_(self.out.weight)
#
#     def forward(self, x0, u_seq):
#         if u_seq.ndim == 2:
#             u_seq = u_seq.unsqueeze(1)
#         B = u_seq.size(0)
#         h0 = self.h0(x0).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
#         c0 = self.c0(x0).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
#         out, _ = self.lstm(u_seq, (h0, c0))
#         # out = self.dropout(out)
#         dx = self.out(out)
#         return x0.unsqueeze(1) + dx



# class QuadLSTM(BaseQuadModel):
#     def __init__(self,
#                  input_dim_u=4,
#                  state_dim_x=12,
#                  hidden_dim=128,
#                  num_layers=2,
#                  dt=0.01):
#         super().__init__(dt)
#         self.input_dim_u = input_dim_u
#         self.state_dim_x = state_dim_x
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#
#         # ---------------------------------------
#         # Structured nonlinear embedding of x0
#         # x = [p(3), v(3), r_log(3), w(3)]
#         # ---------------------------------------
#         embed_dim = 8
#
#         self.embed_p = nn.Sequential(
#             nn.Linear(3, embed_dim),
#             nn.ReLU(),
#         )
#         self.embed_v = nn.Sequential(
#             nn.Linear(3, embed_dim),
#             nn.ReLU(),
#         )
#         self.embed_r = nn.Sequential(
#             nn.Linear(3, embed_dim),
#             nn.ReLU(),
#         )
#         self.embed_w = nn.Sequential(
#             nn.Linear(3, embed_dim),
#             nn.ReLU(),
#         )
#
#         total_embed_dim = 4 * embed_dim
#         self.h0_mlp = nn.Sequential(
#             nn.Linear(total_embed_dim, hidden_dim),
#             nn.Tanh(),
#         )
#         self.c0_mlp = nn.Sequential(
#             nn.Linear(total_embed_dim, hidden_dim),
#             nn.Tanh(),
#         )
#
#         # ---------------------------------------
#         # LSTM over concatenated [x_t, u_t]
#         # ---------------------------------------
#         lstm_input_dim = state_dim_x + input_dim_u
#         self.lstm = nn.LSTM(
#             input_size=lstm_input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#         )
#
#         # Map hidden → delta state (residual)
#         self.out = nn.Linear(hidden_dim, state_dim_x)
#
#         # Small init: near-identity but not dead
#         # nn.init.uniform_(self.out.weight, -1e-4, 1e-4)
#         nn.init.xavier_uniform_(self.out.weight)
#         nn.init.zeros_(self.out.bias)
#
#     # ---- init hidden/cell from x0 ----
#     def _init_states(self, x0):
#         """
#         x0: (B, 12) → h0, c0: (num_layers, B, hidden_dim)
#         """
#         p0 = x0[..., 0:3]
#         v0 = x0[..., 3:6]
#         r0 = x0[..., 6:9]
#         w0 = x0[..., 9:12]
#
#         ep = self.embed_p(p0)
#         ev = self.embed_v(v0)
#         er = self.embed_r(r0)
#         ew = self.embed_w(w0)
#
#         e = torch.cat([ep, ev, er, ew], dim=-1)  # (B, 4*embed_dim)
#
#         h0 = self.h0_mlp(e).unsqueeze(0).repeat(self.num_layers, 1, 1)
#         c0 = self.c0_mlp(e).unsqueeze(0).repeat(self.num_layers, 1, 1)
#         return h0, c0
#
#     def forward(self, x0, u_seq):
#         """
#         x0:    (B, 12)
#         u_seq: (B, T, 4) or (B, T) or (T, 4)
#
#         returns:
#             x_pred_seq: (B, T, 12)
#         """
#         # Ensure batch + time dims
#         if u_seq.ndim == 2:
#             # (T, 4) → (1, T, 4)
#             u_seq = u_seq.unsqueeze(0)
#         if u_seq.ndim != 3:
#             raise ValueError(f"u_seq must be (B,T,4) or (T,4), got {u_seq.shape}")
#
#         B, T, _ = u_seq.shape
#
#         # Init LSTM hidden/cell from x0
#         h, c = self._init_states(x0)
#
#         # Autoregressive rollout in physical state
#         x_t = x0  # (B, 12)
#         preds = []
#
#         for t in range(T):
#             u_t = u_seq[:, t, :]              # (B, 4)
#             lstm_in_t = torch.cat([x_t, u_t], dim=-1)  # (B, 12+4)
#             lstm_in_t = lstm_in_t.unsqueeze(1)         # (B, 1, 16)
#
#             lstm_out_t, (h, c) = self.lstm(lstm_in_t, (h, c))  # (B,1,H)
#             h_t = lstm_out_t[:, 0, :]                          # (B,H)
#
#             dx_t = self.out(h_t)          # (B,12)
#             x_t = x_t + dx_t              # residual dynamics
#
#             preds.append(x_t.unsqueeze(1))
#
#         x_pred_seq = torch.cat(preds, dim=1)  # (B,T,12)
#         return x_pred_seq

class QuadLSTM(nn.Module):
    def __init__(self,
                 input_dim_u=4,
                 state_dim_x=12,
                 hidden_dim=64,
                 num_layers=2,
                 dt=0.01):
        super().__init__()

        self.input_dim_u = input_dim_u
        self.state_dim_x = state_dim_x
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dt = dt

        # Init LSTM state from x0
        self.h0_net = nn.Sequential(
            nn.Linear(state_dim_x, hidden_dim),
            nn.Tanh(),
        )
        self.c0_net = nn.Sequential(
            nn.Linear(state_dim_x, hidden_dim),
            nn.Tanh(),
        )

        # LSTM over controls
        self.lstm = nn.LSTM(
            input_size=input_dim_u,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.post_ln = nn.LayerNorm(hidden_dim)

        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Δx prediction
        self.out = nn.Linear(hidden_dim, state_dim_x)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x0, u_seq):

        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)

        B, T, _ = u_seq.shape

        # Initial LSTM hidden state
        h0 = self.h0_net(x0).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.c0_net(x0).unsqueeze(0).repeat(self.num_layers, 1, 1)

        # LSTM full sequence
        lstm_out, _ = self.lstm(u_seq, (h0, c0))  # (B,T,H)

        # Process hidden state
        lstm_out = self.post_ln(lstm_out)
        h = self.post_mlp(lstm_out)

        # Predict Δx_t for ALL steps
        dx = self.out(h)      # (B,T,12)

        # Compute cumulative sum: x_t = x0 + Σ_{i < t} dx_i
        dx_cumsum = torch.cumsum(dx, dim=1)  # (B,T,12)

        # Vectorized: add x0 to each timestep
        x_pred = x0.unsqueeze(1) + dx_cumsum

        return x_pred   # (B,T,12)


def main():
    model = QuadLSTM(hidden_dim=64, num_layers=1).eval()
    phys_params = {
        "g": 9.81,
        "m": 0.045,
        "J": torch.diag(torch.tensor([2.3951e-5, 2.3951e-5, 3.2347e-6])),
        "thrust_to_weight": 2.0,
        "max_torque": torch.tensor([1e-2, 1e-2, 3e-3]),
    }

    phys_model = PhysQuadModel(phys_params, 0.01)
    neural_model = NeuralQuadModel(hidden_dim=64, num_layers=5, dt=0.01)

    model = ResidualQuadModel(
        phys=phys_model,
        neural=neural_model
    )

    dummy_x0  = torch.randn(1, 12)
    dummy_seq = torch.randn(1, 50, 4)

    # If your model takes (x0, u_seq):
    flops, params = profile(model, inputs=(dummy_x0, dummy_seq))

    print(f"FLOPs: {flops/1e6:.2f}M")
    print(f"Params: {params/1e3:.2f}k")

if __name__ == "__main__":
    main()
