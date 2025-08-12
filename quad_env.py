import math
import numpy as np
import mujoco as mj
import gymnasium as gym
from gymnasium import spaces

# ---------- Low-level MuJoCo-backed quad with rigid payload ----------
class QuadCore:
    def __init__(self, xml_path="quad.xml", dt=0.002, n_motors=4, seed=0):
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.dt = dt
        self.n = n_motors
        self.rng = np.random.default_rng(seed)

        # IDs
        self.base_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base")
        self.payload_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "payload")
        self.payload_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "payload_geom")
        self.rotor_sids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, f"rotor{i}") for i in range(self.n)]

        # Motor internal state (first-order lag) + last command
        self.x = np.zeros(self.n)
        self.u_last = np.zeros(self.n)

        # Thrust/torque constants & lags (randomized on reset)
        self.kT = np.ones(self.n) * 1.8e-4
        self.kQ = np.ones(self.n) * 2.6e-6
        self.tau = np.ones(self.n) * 0.06
        self.spin = np.array([+1, -1, +1, -1])  # CW/CCW pattern
        self.u_min = np.zeros(self.n)
        self.u_max = np.ones(self.n)

        # Aerodynamics
        self.rho = 1.225
        self.CdA = np.array([0.07, 0.07, 0.11])  # slightly higher due to payload
        self.wind = np.zeros(3)

        # Safety box
        self.pos_limit = np.array([2.0, 2.0, 2.0])

        # Hover target
        self.p_ref = np.array([0.0, 0.0, 1.0])

    # ---- helpers ----
    def _R_world_from_body(self):
        return self.data.xmat[self.base_bid].reshape(3, 3)

    def _site_world_pos(self, sid):
        return self.data.site_xpos[sid].copy()

    def _motor_step(self, u_cmd):
        # first-order lag
        self.x += (u_cmd - self.x) * (self.dt / np.maximum(self.tau, 1e-3))
        self.x = np.clip(self.x, 0.0, 1.0)

    def _aero_drag(self):
        v_world = self.data.cvel[self.base_bid, 3:6].copy()
        v_rel = v_world - self.wind
        return -0.5 * self.rho * self.CdA * v_rel * np.abs(v_rel)


    # ---- RL API ----
    def reset(self):
        mj.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.x[:] = 0.0
        self.u_last[:] = 0.0

        # Base mass/inertia (randomized)
        mass_base = self.rng.uniform(0.6, 1.2)  # kg (without payload)
        Jdiag = self.rng.uniform([0.015, 0.015, 0.03], [0.035, 0.035, 0.06])
        self.model.body_mass[self.base_bid] = mass_base
        self.model.body_inertia[self.base_bid] = Jdiag

        # Payload ≈ 1 lb with small variation & vertical offset
        m_payload = self.rng.uniform(0.40, 0.50)     # kg around 0.45
        z_off = -self.rng.uniform(0.10, 0.18)        # 10–18 cm below base
        self.model.body_mass[self.payload_bid] = m_payload
        self.model.body_pos[self.payload_bid] = np.array([0.0, 0.0, z_off])

        # Approximate payload inertia as a small box
        a, b, c = 0.12, 0.12, 0.06
        Ixx = (m_payload * (b*b + c*c)) / 12.0
        Iyy = (m_payload * (a*a + c*c)) / 12.0
        Izz = (m_payload * (a*a + b*b)) / 12.0
        self.model.body_inertia[self.payload_bid] = np.array([Ixx, Iyy, Izz])

        # Motors/env randomization
        self.kT = self.rng.uniform(1.4e-4, 2.2e-4, size=self.n)
        self.kQ = self.rng.uniform(2.0e-6, 3.2e-6, size=self.n)
        self.tau = self.rng.uniform(0.04, 0.09, size=self.n)
        self.wind = self.rng.uniform(-1.0, 1.0, size=3) * self.rng.choice([0.0, 0.5, 1.5])

        # Small pose/attitude jitter
        self.data.qpos[0:3] = np.array([0, 0, 1.0]) + self.rng.normal(scale=0.02, size=3)
        quat = np.array([1, 0, 0, 0]) + self.rng.normal(scale=0.02, size=4)
        quat /= np.linalg.norm(quat)
        self.data.qpos[3:7] = quat

        mj.mj_forward(self.model, self.data)
        return self.obs()

    def obs(self):
        q = self.data.qpos[3:7].copy()
        # world-frame angular & linear velocity from cvel
        w_world = self.data.cvel[self.base_bid, 0:3].copy()
        v_world = self.data.cvel[self.base_bid, 3:6].copy()
        p = self.data.xpos[self.base_bid].copy()
        a = np.zeros(3)
        return np.concatenate([q, w_world, v_world, p, a, self.u_last])


    def step(self, u_cmd):
        u = np.clip(u_cmd, 0.0, 1.0)
        self._motor_step(u)

        # compute forces with current orientation
        R = self._R_world_from_body()
        F_total = np.zeros(3); T_total = np.zeros(3)
        self.data.xfrc_applied[self.base_bid, :6] = 0.0

        axis_world = R[:, 2]
        base_pos = self.data.xpos[self.base_bid].copy()
        for i, sid in enumerate(self.rotor_sids):
            xi = self.x[i]
            T = self.kT[i] * (xi * xi)
            Q = self.kQ[i] * (xi * xi) * self.spin[i]
            Fw = T * axis_world
            rw = self._site_world_pos(sid) - base_pos
            Tw = Q * axis_world + np.cross(rw, Fw)
            F_total += Fw; T_total += Tw

        F_total += self._aero_drag()

        self.data.xfrc_applied[self.base_bid, :3] = F_total
        self.data.xfrc_applied[self.base_bid, 3:] = T_total
        mj.mj_step(self.model, self.data)

        # *** recompute orientation AFTER step for reward ***
        R = self._R_world_from_body()
        z_body_world = R[:, 2]

        p = self.data.xpos[self.base_bid].copy()
        v = self.data.cvel[self.base_bid, 3:6].copy()
        w = self.data.cvel[self.base_bid, 0:3].copy()

        angle_err = math.acos(float(np.clip(z_body_world @ np.array([0, 0, 1]), -1.0, 1.0)))
        pos_err = np.linalg.norm(p - self.p_ref)
        vel_err = np.linalg.norm(v)
        rate_err = np.linalg.norm(w)
        du = np.linalg.norm(u - self.u_last)

        r = -2.0*angle_err -1.5*pos_err -0.5*vel_err -0.2*rate_err -0.05*du + 0.01
        self.u_last = u.copy()

        done = (
            np.any(np.abs(p[:2]) > self.pos_limit[:2]) or
            (p[2] < 0.2 or p[2] > self.pos_limit[2]) or
            (angle_err > math.radians(60))
        )
        return self.obs(), r, done, {}

# ---------- Gymnasium wrapper (so you can plug into SB3/CleanRL) ----------
class MujocoQuadEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, xml_path="quad.xml", seed=0):
        super().__init__()
        self.core = QuadCore(xml_path=xml_path, seed=seed)
        self.n = self.core.n

        # Obs: [quat(4), w(3), v(3), p(3), a(3), last_u(n)]
        obs_dim = 4 + 3 + 3 + 3 + 3 + self.n
        high_obs = np.ones(obs_dim, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        # Actions are normalized ESC commands per motor in [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        obs = self.core.reset().astype(np.float32)
        return obs, {}

    def step(self, action):
        obs, r, done, info = self.core.step(np.asarray(action, dtype=np.float32))
        return obs.astype(np.float32), float(r), done, False, info

# ---------- quick smoke test ----------
if __name__ == "__main__":
    env = MujocoQuadEnv("quad.xml", seed=42)
    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    for t in range(4000):
        # crude hover guess; your RL agent will replace this
        u = np.clip(np.ones(env.n) * 0.60 + 0.02 * np.random.randn(env.n), 0, 1)
        obs, r, terminated, truncated, _ = env.step(u)
        ep_ret += r; ep_len += 1
        if terminated or truncated:
            print(f"Episode: return={ep_ret:.2f} len={ep_len}")
            obs, _ = env.reset(); ep_ret, ep_len = 0.0, 0
