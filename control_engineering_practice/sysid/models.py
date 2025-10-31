import torch
import torch.nn as nn
import numpy as np

# class PhysQuadModel(nn.Module):
#     def __init__(self, params, dt: float):
#         super().__init__()
#         self.dt = dt
#
#         # Physical constants
#         self.g = params["g"]
#         self.m = params["m"]
#         self.J = torch.tensor(params["J"], dtype=torch.float32)   # [3,3]
#         self.thrust_to_weight = params["thrust_to_weight"]
#         self.max_torque = torch.tensor(params["max_torque"], dtype=torch.float32)  # [3]
#
#         self.gravity_vec = torch.tensor([0.0, 0.0, self.m * self.g], dtype=torch.float32)
#
#     def forward(self, x, u):
#         """
#         One Euler step of the dynamics.
#         x: [batch, 13] = [pos(3), vel(3), quat(4), omega(3)]
#         u: [batch, 4]  = [T_norm, τx_norm, τy_norm, τz_norm]
#         returns: x_next [batch, 13]
#         """
#         dx = self.compute_dx(x, u)
#         x_next = x + self.dt * dx
#         return x_next, dx
#
#     def compute_dx(self, x, u):
#         device, dtype = x.device, x.dtype
#
#         pos   = x[:, 0:3]
#         vel   = x[:, 3:6]
#         quat  = x[:, 6:10]   # [x,y,z,w]
#         omega = x[:, 10:13]
#
#         # Saturate controls
#         T_norm   = torch.clamp(u[:, 0], 0.0, 1.0)
#         tau_norm = torch.clamp(u[:, 1:], -1.0, 1.0)
#
#         # Convert to physical units
#         T_max = self.thrust_to_weight * self.m * self.g
#         T     = T_norm * T_max
#         tau   = tau_norm * self.max_torque.to(device)
#
#         # Translational dynamics
#         thrust_b = torch.stack([
#             torch.zeros_like(T),
#             torch.zeros_like(T),
#             T
#         ], dim=1)  # [B,3]
#
#         thrust_world = self.quat_rotate(quat, thrust_b)
#         acc = (thrust_world - self.gravity_vec.to(device)) / self.m
#
#         # Rotational dynamics
#         J = self.J.to(device)
#         omega_cross = torch.cross(omega, torch.matmul(omega, J.T), dim=-1)
#         tau_minus_cross = tau - omega_cross
#         omega_dot = torch.linalg.solve(J, tau_minus_cross.T).T
#
#         # Quaternion kinematics
#         quat_dot = self.quat_derivative(quat, omega)
#
#         # Assemble derivative
#         dx = torch.cat([vel, acc, quat_dot, omega_dot], dim=-1)
#         return dx
#
#     # === Quaternion utilities (scalar-last: [x,y,z,w]) ===
#
#     def quat_multiply(self, q1, q2):
#         # Extract scalar-last but unpack like JAX does (w first)
#         w1, x1, y1, z1 = q1[:, 3], q1[:, 0], q1[:, 1], q1[:, 2]
#         w2, x2, y2, z2 = q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]
#
#         x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#         y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#         z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
#         w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#
#         return torch.stack([x, y, z, w], dim=-1)
#
#     def quat_rotate(self, q, v):
#         # v: [B,3] → quaternion with zero scalar part
#         q_conj = torch.cat([-q[:, :3], q[:, 3:]], dim=-1)
#         v_quat = torch.cat([v, torch.zeros(v.shape[0], 1, device=v.device)], dim=-1)
#         temp = self.quat_multiply(q, v_quat)
#         rotated = self.quat_multiply(temp, q_conj)
#         return rotated[:, :3]
#
#     def quat_derivative(self, q, omega):
#         omega_quat = torch.cat([omega, torch.zeros(omega.shape[0], 1, device=omega.device)], dim=-1)
#         dq = 0.5 * self.quat_multiply(q, omega_quat)
#         return dq

class PhysQuadModel(nn.Module):
    """
    Optimized differentiable quadrotor physics model.
    Vectorized, GPU-friendly, and free of per-step allocations.
    """

    def __init__(self, params, dt: float, x_scaler=None):
        super().__init__()
        self.dt = dt

        # === Physical constants ===
        self.g = params["g"]
        self.m = params["m"]
        self.thrust_to_weight = params["thrust_to_weight"]

        # Register constants as non-trainable buffers (faster + always on same device)
        self.register_buffer("J", torch.tensor(params["J"], dtype=torch.float32))  # [3,3]
        self.register_buffer("J_inv", torch.linalg.inv(self.J))                     # precompute inverse
        self.register_buffer("max_torque", torch.tensor(params["max_torque"], dtype=torch.float32))
        self.register_buffer("gravity_vec", torch.tensor([0.0, 0.0, self.g], dtype=torch.float32))

    # =====================================================
    # --- Forward Dynamics ---
    # =====================================================
    def forward(self, x, u):
        """
        One Euler integration step.
        x: [B, 13] = [pos(3), vel(3), quat(4), omega(3)]
        u: [B, 4]  = [T_norm, τx_norm, τy_norm, τz_norm]
        returns: x_next, dx
        """
        dx = self.compute_dx(x, u)
        x_next = x + self.dt * dx
        return x_next, dx

    def scale(self, x):
        return (x - self.x_mean) / self.x_std

    def unscale(self, x):
        return x * self.x_std + self.x_mean

    # =====================================================
    # --- Compute time derivative of state ---
    # =====================================================
    def compute_dx(self, x, u):
        pos = x[:, 0:3]
        vel = x[:, 3:6]
        quat = x[:, 6:10]
        omega = x[:, 10:13]

        T_norm = torch.clamp(u[:, 0], 0.0, 1.0)
        tau_norm = torch.clamp(u[:, 1:], -1.0, 1.0)

        T_max = self.thrust_to_weight * self.m * self.g
        T = T_norm * T_max
        tau = tau_norm * self.max_torque

        thrust_b = torch.zeros_like(omega)
        thrust_b[:, 2] = T
        thrust_world = self.quat_rotate_fast(quat, thrust_b)

        # add wind_force if applicable
        wind_force = getattr(self, "wind_force", torch.zeros_like(thrust_world))
        acc = (thrust_world + wind_force - self.m * self.gravity_vec) / self.m

        omega_cross = torch.cross(omega, omega @ self.J, dim=-1)
        omega_dot = torch.linalg.solve(self.J, (tau - omega_cross).unsqueeze(-1)).squeeze(-1)

        quat_dot = self.quat_derivative_fast(quat, omega)
        dx = torch.cat([vel, acc, quat_dot, omega_dot], dim=-1)
        return dx

    # =====================================================
    # --- Fast Quaternion Math (scalar-last) ---
    # =====================================================
    @staticmethod
    def quat_rotate_fast(q, v):
        """
        Rotate vector v (B,3) by quaternion q (B,4) [x,y,z,w].
        Equivalent to q * [v,0] * q_conj but much faster.
        """
        q_vec = q[:, :3]
        q_w = q[:, 3:4]
        # v' = v + 2 * cross(q_vec, cross(q_vec, v) + q_w * v)
        qv = torch.cross(q_vec, v, dim=-1) + q_w * v
        return v + 2.0 * torch.cross(q_vec, qv, dim=-1)

    @staticmethod
    def quat_derivative_fast(q, omega):
        """
        dq/dt = 0.5 * q ⊗ [ω, 0]
        Vectorized direct formula, avoids two quaternion multiplications.
        """
        q_vec = q[:, :3]
        q_w = q[:, 3:4]
        dq_vec = 0.5 * (q_w * omega + torch.cross(q_vec, omega, dim=-1))
        dq_w = -0.5 * (q_vec * omega).sum(dim=-1, keepdim=True)
        return torch.cat([dq_vec, dq_w], dim=-1)


class MotorsPhysQuadModel(nn.Module):
    """
    Quadrotor model that receives motor angular speeds (rad/s or ERPM)
    and converts them to normalized thrust/torques using the Mellinger structure.
    Two learnable parameters: Kt (thrust, roll, pitch) and Kc (yaw coupling).
    """
    def __init__(self, base_model: PhysQuadModel):
        super().__init__()
        self.base_model = base_model
        self.arm_length = 0.04
        self.dt = base_model.dt

        # === Learnable parameters (small init) ===
        self.Kt = 3.788e-08  # thrust / roll / pitch gain
        self.Kc = 3.788e-11
        # self.act = nn.Tanh()  # keeps values in [-1, 1]

        # === Physical constants from base model ===
        self.m = base_model.m
        self.g = base_model.g
        self.thrust_to_weight = base_model.thrust_to_weight
        self.max_torque = base_model.max_torque  # tensor([τx_max, τy_max, τz_max])

        # Precompute T_max (scalar)
        self.T_max = self.thrust_to_weight * self.m * self.g


    def forward(self, x, u_motors):
        """
        x: [B, 13] - state vector
        u_motors: [B, 4] - motor angular speeds (ERPM or rad/s)
        returns: x_next, dx
        """
        u2 = u_motors ** 2  # [B, 4], since thrust ∝ ω²

        # === Compute thrust and torques ===
        T = self.Kt * (u2[:, 0] + u2[:, 1] + u2[:, 2] + u2[:, 3])
        tau_x  = self.Kt * self.arm_length * ((u2[:, 2] + u2[:, 3]) - (u2[:, 0] + u2[:, 1]))
        tau_y  = self.Kt * self.arm_length * ((u2[:, 1] + u2[:, 2]) - (u2[:, 0] + u2[:, 3]))
        tau_z  = self.Kc * ((u2[:, 0] + u2[:, 2]) - (u2[:, 1] + u2[:, 3]))

        # --- Normalize according to PhysQuadModel conventions ---
        T_norm   = T / self.T_max
        tau_norm = torch.stack([tau_x, tau_y, tau_z], dim=1) / self.max_torque  # elementwise division

        # Combine into normalized control vector [T_norm, τx_norm, τy_norm, τz_norm]
        u_raw = torch.cat([T_norm.unsqueeze(1), tau_norm], dim=1)
        u_phys = u_raw

        # === Pass to physics model ===
        x_next, dx = self.base_model(x, u_phys)
        return x_next, dx


# class MotorsPhysQuadModel(nn.Module):
#     """
#     Quadrotor model that receives motor angular speeds (rad/s or ERPM)
#     and converts them to normalized thrust/torques using the Mellinger structure.
#     Two learnable parameters: Kt (thrust, roll, pitch) and Kc (yaw coupling).
#     """
#
#     def __init__(self, base_model: PhysQuadModel):
#         super().__init__()
#         self.base_model = base_model
#         self.arm_length = 0.05
#         self.dt = base_model.dt
#         self.fc = nn.Linear(4, 4)
#         nn.init.zeros_(self.fc.weight)
#         self.fc.bias.data.fill_(0.0)
#         self.act = nn.Tanh()  # keeps values in [-1, 1]
#
#     def forward(self, x, u_motors):
#         u_raw = self.fc(u_motors)
#         u_phys = self.act(u_raw)
#         u_phys = torch.cat([(u_phys[:, 0:1] + 1) / 2, u_phys[:, 1:]], dim=1)
#         # === Pass to physics model ===
#         x_next, dx = self.base_model(x, u_phys)
#         return x_next, dx

class NeuralQuadModel(nn.Module):
    def __init__(self, dt=0.01, state_dim=13, input_dim=4, hidden_dim=1024, num_layers=6, layer_norm=False):
        super().__init__()
        self.dt = dt
        self.state_dim = state_dim
        self.input_dim = input_dim

        layers = []
        in_dim = state_dim + input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, state_dim)

    def forward(self, x, u):
        """
        x: [B, state_dim]
        u: [B, input_dim]
        Returns x_next_pred: [B, state_dim]
        """
        xu = torch.cat([x, u], dim=-1)
        h = self.mlp(xu)
        dx = self.out(h)
        # dx = self.scale_output(dx)  # scale residual step for stability
        x_next = x + dx
        return x_next, dx  # predict Δx for better learning stability


class ResidualQuadModel(nn.Module):
    def __init__(self, phys_model, neural_model):
        super().__init__()
        self.phys_model = phys_model
        self.neural_model = neural_model
        assert hasattr(phys_model, "dt") and hasattr(neural_model, "dt"), \
            "Both models must define dt."
        assert phys_model.dt == neural_model.dt, "dt mismatch between models"
        self.dt = phys_model.dt

    def forward(self, x, u):
        with torch.no_grad():
            _, dx_phys = self.phys_model(x, u)
        _, dx_neur = self.neural_model(x, u)

        # Combine derivatives, then integrate once
        x_next = x + self.dt * (dx_phys + dx_neur)

        return x_next


class QuadMultiStepModel(nn.Module):
    """
    Multi-step rollout model for quadrotor dynamics.
    Supports three operation modes:
        - "physics": use only physics-based model
        - "neural": use only neural model
        - "residual": combine both (x_{t+1} = x_t + dt * (dx_phys + dx_neural))
    """

    def __init__(self, phys_model=None, neural_model=None, mode="residual"):
        super().__init__()

        self.phys_model = phys_model
        self.neural_model = neural_model
        self.mode = mode.lower()

        assert self.mode in ["physics", "neural", "residual"], \
            f"Invalid mode '{mode}'. Choose from ['physics', 'neural', 'residual']."

        # Check dt consistency and availability
        if self.mode == "physics":
            assert phys_model is not None, "Physics model required for 'physics' mode."
            self.dt = phys_model.dt

        elif self.mode == "neural":
            assert neural_model is not None, "Neural model required for 'neural' mode."
            self.dt = neural_model.dt

        elif self.mode == "residual":
            assert phys_model is not None and neural_model is not None, \
                "Both models required for 'residual' mode."
            assert hasattr(phys_model, "dt") and hasattr(neural_model, "dt")
            assert abs(phys_model.dt - neural_model.dt) < 1e-9, "dt mismatch between models"
            self.dt = phys_model.dt

    # ------------------------------------------------------
    # One-step forward depending on selected mode
    # ------------------------------------------------------
    def one_step(self, x, u):
        """
        Single-step prediction.
        x: [B, state_dim]
        u: [B, input_dim]
        """
        if self.mode == "physics":
            if x.ndim == 3:
                x = x.squeeze(1)  # (B, 13)
            x_next, _ = self.phys_model(x, u)
            return x_next

        elif self.mode == "neural":
            x_next, _ = self.neural_model(x, u)
            return x_next

        elif self.mode == "residual":
            _, dx_phys = self.phys_model(x, u)
            _, dx_neur = self.neural_model(x, u)
            x_next = x + (dx_phys + dx_neur)
            return x_next

    # ------------------------------------------------------
    # Multi-step rollout
    # ------------------------------------------------------
    def forward(self, x0, u_seq):
        """
        Multi-step rollout.

        Args:
            x0:     [B, 13]          initial state(s)
            u_seq:  [B, H, 4]        control sequence (horizon H)

        Returns:
            x_preds: [B, H, 13]      predicted trajectory
        """
        if u_seq.dim() == 2:
            # Assume [H, 4] → add batch dimension
            u_seq = u_seq.unsqueeze(0)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)

        batch_size, horizon, _ = u_seq.shape
        x_preds = []
        x = x0

        for t in range(horizon):
            u_t = u_seq[:, t, :]
            x_next = self.one_step(x, u_t)
            x_preds.append(x_next.unsqueeze(1))
            x = x_next  # feed back predicted state

        x_preds = torch.cat(x_preds, dim=1)
        return x_preds

class QuadLSTM(nn.Module):
    def __init__(self, input_dim_u=4, state_dim_x=12, hidden_dim=12, num_layers=5):
        """
        LSTM model conditioned on initial state x0.

        Receives a control input sequence (u0, …, u_{N−1})
        and an initial state x0, and predicts the corresponding
        state sequence (x1, …, xN).

        Args:
            input_dim_u: dimension of control input u_t
            state_dim_x: dimension of system state x_t
            hidden_dim:  hidden size of the LSTM
            num_layers:  number of stacked LSTM layers
        """
        super().__init__()
        self.dt = 0.01
        self.input_dim_u = input_dim_u
        self.state_dim_x = state_dim_x
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Core LSTM block
        self.lstm = nn.LSTM(
            input_size=input_dim_u,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Linear mappings from x0 → (h0, c0)
        self.h0_layer = nn.Linear(state_dim_x, hidden_dim)
        self.c0_layer = nn.Linear(state_dim_x, hidden_dim)

        # Output layer: hidden → predicted state
        self.output_layer = nn.Linear(hidden_dim, state_dim_x)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)


    def forward(self, x0, u_seq):
        """
        Args:
            u_seq: (B, N, input_dim_u) control sequence
            x0:    (B, state_dim_x) initial state

        Returns:
            x_pred_seq: (B, N, state_dim_x) predicted state sequence
        """
        B = u_seq.size(0)

        # Project x0 → initial hidden and cell states
        h0 = self.h0_layer(x0).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
        c0 = self.c0_layer(x0).unsqueeze(0).repeat(self.num_layers, 1, 1)

        # Run LSTM
        lstm_out, _ = self.lstm(u_seq, (h0, c0))  # (B, N, hidden_dim)

        # Predict residual states (Δx)
        dx_seq = self.output_layer(lstm_out)  # (B, N, state_dim_x)

        # Add initial state x0 to get absolute prediction
        x_pred_seq = x0.unsqueeze(1) + dx_seq  # broadcast addition

        return x_pred_seq

# class QuadLSTM(nn.Module):
#     def __init__(self, input_dim_u=4, state_dim_x=12, hidden_dim=12, num_layers=5):
#         """
#         LSTM model conditioned on initial state x0.
#
#         Receives a control input sequence (u0, …, u_{N−1})
#         and an initial state x0, and predicts the corresponding
#         state sequence (x1, …, xN).
#
#         Args:
#             input_dim_u: dimension of control input u_t
#             state_dim_x: dimension of system state x_t
#             hidden_dim:  hidden size of the LSTM
#             num_layers:  number of stacked LSTM layers
#         """
#         super().__init__()
#         self.dt = 0.01
#         self.input_dim_u = input_dim_u
#         self.state_dim_x = state_dim_x
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#
#         # Core LSTM block
#         self.lstm = nn.LSTM(
#             input_size=input_dim_u,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True
#         )
#
#         # Linear mappings from x0 → (h0, c0)
#         self.h0_layer = nn.Linear(state_dim_x, hidden_dim)
#         self.c0_layer = nn.Linear(state_dim_x, hidden_dim)
#
#         # Output layer: hidden → predicted state
#         self.output_layer = nn.Linear(hidden_dim, state_dim_x)
#         nn.init.zeros_(self.output_layer.weight)
#         nn.init.zeros_(self.output_layer.bias)
#
#         self._identity_lstm_init()
#
#     def _identity_lstm_init(self):
#         for name, param in self.lstm.named_parameters():
#             if "weight" in name:
#                 nn.init.zeros_(param)
#             elif "bias" in name:
#                 # Each LSTM bias vector has [b_i | b_f | b_g | b_o]
#                 n = param.shape[0] // 4
#                 param.data.fill_(0)
#                 # forget gate bias to +5, input gate to -5, output gate to +5
#                 param.data[n:2 * n] = 5.0  # forget gate bias
#                 param.data[0:n] = -5.0  # input gate bias
#                 param.data[3 * n:4 * n] = 5.0  # output gate bias
#
#     def forward(self, x0, u_seq):
#         """
#         Args:
#             u_seq: (B, N, input_dim_u) control sequence
#             x0:    (B, state_dim_x) initial state
#
#         Returns:
#             x_pred_seq: (B, N, state_dim_x) predicted state sequence
#         """
#         B = u_seq.size(0)
#
#         # Project x0 → initial hidden and cell states
#         # h0 = self.h0_layer(x0).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
#         # c0 = self.c0_layer(x0).unsqueeze(0).repeat(self.num_layers, 1, 1)
#
#         h0 = x0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, state_dim)
#         c0 = torch.zeros_like(h0)
#
#         # Run LSTM
#         lstm_out, _ = self.lstm(u_seq, (h0, c0))  # (B, N, hidden_dim)
#
#         # Predict residual states (Δx)
#         # dx_seq = self.output_layer(lstm_out)  # (B, N, state_dim_x)
#         dx_seq = lstm_out
#
#         # Add initial state x0 to get absolute prediction
#         x_pred_seq = x0.unsqueeze(1) + dx_seq  # broadcast addition
#
#         return x_pred_seq


class NeuralQuadMultistepModel(nn.Module):
    def __init__(self, dt=0.01, state_dim=12, input_dim=4, hidden_dim=1024, num_layers=6, layer_norm=False):
        super().__init__()
        self.dt = dt
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_norm = layer_norm

        layers = []
        in_dim = state_dim + input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, state_dim)

        # === Zero-initialize all parameters so model == identity (naive baseline) ===
        self._zero_init()

    def _zero_init(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if name == "out":
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x0, u_seq):
        """
        Multi-step rollout.
        Args:
            x0:    [B, state_dim]
            u_seq: [B, H, input_dim]
        Returns:
            x_preds: [B, H, state_dim]
        """
        if u_seq.dim() == 2:
            u_seq = u_seq.unsqueeze(0)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)

        batch_size, horizon, _ = u_seq.shape
        x_preds = []
        x_t = x0

        for t in range(horizon):
            u_t = u_seq[:, t, :]
            inp = torch.cat([x_t, u_t], dim=-1)
            mlp_out = self.mlp(inp)
            x_next = x_t + self.out(mlp_out)  # residual update
            x_preds.append(x_next.unsqueeze(1))
            x_t = x_next  # feed next state

        x_preds = torch.cat(x_preds, dim=1)
        return x_preds

class QuadLSTMModular(nn.Module):
    def __init__(self, input_dim_u=4, state_dim_x=3, hidden_dim=13, num_layers=5):
        """
        LSTM model conditioned on initial state x0.

        Receives a control input sequence (u0, …, u_{N−1})
        and an initial state x0, and predicts the corresponding
        state sequence (x1, …, xN).

        Args:
            input_dim_u: dimension of control input u_t
            state_dim_x: dimension of system state x_t
            hidden_dim:  hidden size of the LSTM
            num_layers:  number of stacked LSTM layers
        """
        super().__init__()

        # 🔹 Learnable dt parameter (positive scalar)
        self.log_dt = nn.Parameter(torch.log(torch.tensor(1e-4)))  # store log(dt) for positivity constraint
        self.input_dim_u = input_dim_u
        self.state_dim_x = state_dim_x
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Core LSTM block
        self.lstm = nn.LSTM(
            input_size=input_dim_u,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Linear mappings from x0 → (h0, c0)
        self.h0_layer = nn.Linear(state_dim_x, hidden_dim)
        self.c0_layer = nn.Linear(state_dim_x, hidden_dim)

        # Output layer: hidden → predicted state
        self.output_layer = nn.Linear(hidden_dim, state_dim_x)

    @property
    def dt(self):
        """Return positive time-step value."""
        return torch.exp(self.log_dt)

    def forward(self, x0, u_seq):
        """
        Args:
            u_seq: (B, N, input_dim_u) control sequence
            x0:    (B, state_dim_x) initial state

        Returns:
            x_pred_seq: (B, N, state_dim_x) predicted state sequence
        """
        B = u_seq.size(0)

        # Project x0 → initial hidden and cell states
        h0 = self.h0_layer(x0).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
        c0 = self.c0_layer(x0).unsqueeze(0).repeat(self.num_layers, 1, 1)

        # Run LSTM
        lstm_out, _ = self.lstm(u_seq, (h0, c0))  # (B, N, hidden_dim)

        # Map to predicted states
        x_pred_seq = self.output_layer(lstm_out)  # (B, N, state_dim_x)

        return x_pred_seq

