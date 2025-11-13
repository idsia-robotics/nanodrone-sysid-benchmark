import jax
import jax.numpy as jnp
import diffrax as dx
import numpy as np
from functools import partial

def step_dynamics_euler(x, u, dt, dynamics_fn, params):
    """One Euler integration step for quaternion dynamics"""
    dx = dynamics_fn(x, u, params)
    
    x_next = x + dx * dt
    
    # Re-normalize quaternion to avoid drift
    quat_next = x_next[6:10]
    quat_next /= jnp.linalg.norm(quat_next)
    x_next = x_next.at[6:10].set(quat_next)
    
    return x_next

def step_dynamics_rk4(x, u, dt, dynamics_fn, params):
    f = lambda x_: dynamics_fn(x_, u, params)

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)

    x_next = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)

    # Re-normalize quaternion to avoid drift
    quat_next = x_next[6:10]
    quat_next /= jnp.linalg.norm(quat_next)
    x_next = x_next.at[6:10].set(quat_next)
    
    return x_next

def step_dynamics_dopri5(x, u, dt, dynamics_fn, params):
    # f = lambda _, x_, u_: dynamics_fn(x_, u_, params)
    def f(_, x_, u_): 
        return dynamics_fn(x_, u_, params)
    term = dx.ODETerm(f)

    solver = dx.Dopri5()
    sol = dx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=dt,
        dt0=dt/10,
        y0=x,
        args=u,
        saveat=dx.SaveAt(t1=True),
    )
    x_next = sol.ys[0]

    return x_next

@partial(jax.jit, static_argnames=['dt', 'dynamics_fn', 'step_fn'])
def simulate_rollout(x0, U, dt, dynamics_fn, params, step_fn=step_dynamics_dopri5):
    """
    Simulate a rollout starting from x0 using a given integration function.

    Parameters:
    - x0: initial state, shape (13,)
    - U: control inputs, shape (T, 4)
    - dt: time step
    - params: dynamics parameters
    - step_fn: integration function, e.g., step_dynamics_rk4

    Returns:
    - X: full state trajectory, shape (T+1, 13), includes x0
    """
    def body(x, u):
        x_next = step_fn(x, u, dt, dynamics_fn, params)
        return x_next, x_next

    _, X = jax.lax.scan(body, x0, U)

    # Prepend x0
    X = jnp.concatenate([x0[None, :], X])
    
    return X

@partial(jax.jit, static_argnames=['t1', 'dt', 'setpoint_fn', 'controller_fn', 'dynamics_fn', 'step_fn'])
def simulate_closed_loop(x0, t1, dt, setpoint_fn, controller_fn, dynamics_fn, sys_params, ctrl_params, step_fn=step_dynamics_dopri5):
    """
    Simulate a closed-loop trajectory with feedback controller.

    Args:
        x0: initial state, shape (13,)
        setpoint_fn: function(t: float) -> setpoint dict
        controller_fn: function(x, x_d, sys_params, ctrl_params) -> u [4]
        sys_params: dynamics parameters
        ctrl_params: controller parameters
        dt: time step [s]
        T: number of steps
        dynamics_fn: function (x, u, sys_params) -> dx/dt
        step_fn: integration function (e.g. step_dynamics_rk4)

    Returns:
        X: state trajectory, shape (T+1, 13)
        U: control inputs, shape (T, 4)
    """

    def body(carry, inputs):
        x, ctrl_state = carry
        t = inputs

        with jax.profiler.TraceAnnotation("setpoint_step"):
            x_d = setpoint_fn(t)
        
        with jax.profiler.TraceAnnotation("ctrl_step"):
            u, ctrl_state = controller_fn(x, x_d, sys_params, ctrl_params, ctrl_state)
            
        with jax.profiler.TraceAnnotation("sim_step"):
            x_next = step_fn(x, u, dt, dynamics_fn, sys_params)

        carry = x_next, ctrl_state
        out = x, u, x_d
        return carry, out
    
    t0 = 0
    duration = t1 - t0
    N = np.ceil(duration / dt).astype(int)
    t = jnp.linspace(t0, t1, N)
    
    initial_carry = (x0, None)
    inputs = t
    initial_carry, _ = body(initial_carry, inputs[0]) # FIXME: is this really needed?
    
    _, outputs = jax.lax.scan(body, initial_carry, inputs)
    X, U, Xd = outputs

    # Prepend x0
    X  = jnp.concatenate([x0[None, :], X])
 
    return X, U, Xd

