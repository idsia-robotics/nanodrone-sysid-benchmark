import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, params, dt, arm_length=0.04, Kt=3.8e-08, Kc=3.8e-11):
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
        τz = self.Kc * ((ω2[:, 0] + ω2[:, 2]) - (ω2[:, 1] + ω2[:, 3]))
        T_norm = T / self.T_max
        τ_norm = torch.stack([τx, τy, τz], dim=1) / self.max_torque  # (B,3)
        return torch.cat([T_norm.unsqueeze(1), τ_norm], dim=1)  # (B,4)

    def one_step(self, x, u_mot):
        """x: (B,12), u_mot: (B,4) rad/s"""
        u_phys = self.motor_to_phys(u_mot)  # (B,4) normalized physics inputs
        return self._step_from_phys(x, u_phys)

    def _step_from_phys(self, x, u_phys):
        """
        RK4 integration of quadrotor rigid-body dynamics with quaternion attitude.
        State: [pos(3), vel(3), so3(3), omega(3)]
        """

        dt = self.dt

        def f(x, u):
            """Compute state derivatives given current state and controls."""
            pos = x[:, 0:3]
            vel = x[:, 3:6]
            so3 = x[:, 6:9]
            omega = x[:, 9:12]

            quat = self.so3_log_to_quat(so3)

            # --- controls ---
            T_norm = torch.clamp(u[:, 0], 0.0, 1.0)
            tau_norm = torch.clamp(u[:, 1:], -1.0, 1.0)
            T = T_norm * (self.thrust_to_weight * self.m * self.g)
            tau = tau_norm * self.max_torque

            # --- translation ---
            thrust_b = torch.zeros_like(vel)
            thrust_b[:, 2] = T
            thrust_w = self.quat_rotate(quat, thrust_b)
            acc = (thrust_w - self.m * self.gravity) / self.m

            # --- rotation dynamics ---
            Jω = omega @ self.J.T
            omega_dot = torch.linalg.solve(
                self.J, (tau - torch.cross(omega, Jω, dim=-1)).unsqueeze(-1)
            ).squeeze(-1)

            # --- quaternion derivative ---
            quat_dot = self.quat_derivative(quat, omega)

            return vel, acc, quat_dot, omega_dot

        # === RK4 stages ===
        # convert so3 -> quat for integration
        so3 = x[:, 6:9]
        quat = self.so3_log_to_quat(so3)
        pos = x[:, 0:3]
        vel = x[:, 3:6]
        omega = x[:, 9:12]

        def pack(pos, vel, quat, omega):
            so3 = self.quat_to_so3_log(F.normalize(quat, dim=-1))
            return torch.cat([pos, vel, so3, omega], dim=-1)

        # k1
        v1, a1, qd1, w1 = f(x, u_phys)

        # k2
        pos2 = pos + 0.5 * dt * v1
        vel2 = vel + 0.5 * dt * a1
        quat2 = F.normalize(quat + 0.5 * dt * qd1, dim=-1)
        omega2 = omega + 0.5 * dt * w1
        x2 = pack(pos2, vel2, quat2, omega2)
        v2, a2, qd2, w2 = f(x2, u_phys)

        # k3
        pos3 = pos + 0.5 * dt * v2
        vel3 = vel + 0.5 * dt * a2
        quat3 = F.normalize(quat + 0.5 * dt * qd2, dim=-1)
        omega3 = omega + 0.5 * dt * w2
        x3 = pack(pos3, vel3, quat3, omega3)
        v3, a3, qd3, w3 = f(x3, u_phys)

        # k4
        pos4 = pos + dt * v3
        vel4 = vel + dt * a3
        quat4 = F.normalize(quat + dt * qd3, dim=-1)
        omega4 = omega + dt * w3
        x4 = pack(pos4, vel4, quat4, omega4)
        v4, a4, qd4, w4 = f(x4, u_phys)

        # === integrate ===
        pos_next = pos + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        vel_next = vel + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        omega_next = omega + (dt / 6.0) * (w1 + 2 * w2 + 2 * w3 + w4)
        quat_next = quat + (dt / 6.0) * (qd1 + 2 * qd2 + 2 * qd3 + qd4)
        quat_next = F.normalize(quat_next, dim=-1)

        # convert back to so3
        so3_next = self.quat_to_so3_log(quat_next)

        # final state
        x_next = torch.cat([pos_next, vel_next, so3_next, omega_next], dim=-1)
        return x_next

    # ======================================================
    # === Quaternion utilities ===
    # ======================================================
    @staticmethod
    def quat_to_so3_log(q, eps=1e-6):
        v = q[..., :3]
        w = q[..., 3:]
        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        angle = 2 * torch.atan2(norm_v, w.clamp(min=-1.0 + eps, max=1.0 - eps))
        small = norm_v < eps
        log_q = angle / (norm_v + eps) * v
        log_q = torch.where(small, 2.0 * v, log_q)
        return log_q

    @staticmethod
    def so3_log_to_quat(r):
        theta = torch.linalg.norm(r, dim=-1, keepdim=True)
        small = theta < 1e-6
        half = 0.5 * theta
        v = torch.sin(half) / (theta + 1e-8) * r
        w = torch.cos(half)
        v = torch.where(small, 0.5 * r, v)
        w = torch.where(small, torch.ones_like(w), w)
        return torch.cat([v, w], dim=-1)

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

class QuadLSTM(BaseQuadModel):
    def __init__(self, input_dim_u=4, state_dim_x=12, hidden_dim=64, num_layers=2, dt=0.01):
        super().__init__(dt)
        self.input_dim_u = input_dim_u
        self.state_dim_x = state_dim_x
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim_u, hidden_dim, num_layers, batch_first=True)
        self.h0 = nn.Linear(state_dim_x, hidden_dim)
        self.c0 = nn.Linear(state_dim_x, hidden_dim)
        self.out = nn.Linear(hidden_dim, state_dim_x)
        nn.init.zeros_(self.out.weight)

    def forward(self, x0, u_seq):
        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)
        B = u_seq.size(0)
        h0 = self.h0(x0).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = self.c0(x0).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        out, _ = self.lstm(u_seq, (h0, c0))
        dx = self.out(out)
        return x0.unsqueeze(1) + dx
