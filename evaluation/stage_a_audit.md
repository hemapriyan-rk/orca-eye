# Stage-A Implementation Technical Audit

This audit documents the technical reality of the Stage-A implementation in `D:\orca` as of September 2026. Every equation, data structure, and logical flow is mapped to the corresponding source code.

---

## A. How Bearing $\phi$ is Calculated

- **Source**: [`navigation/geometry.py`](file:///d:/orca/navigation/geometry.py#L31-L47)
- **Mathematical Definition**:
  For an entity with egocentric lateral position $x$ (meters, positive right) and forward ground distance $y$ (meters, positive forward):
  $$\phi = \text{atan2}(x, \max(y, 10^{-6})) \in [-\pi, \pi]$$
  - $\phi = 0$: directly ahead along the forward heading axis (+y).
  - $\phi > 0$: wearer's right flank.
  - $\phi < 0$: wearer's left flank.
- **Degrees Conversion**: $\phi_{\text{deg}} = \frac{180}{\pi} \cdot \phi$.
- **Camera Calibration Mapping**: In [`perception/calibration.py`](file:///d:/orca/perception/calibration.py#L82), for optical principal point $(c_x, c_y)$ and focal length $f_x$, an image point at pixel column $u$ projects to a ground bearing:
  $$\phi_{\text{ray}} = \text{atan2}\left(\frac{u - c_x}{f_x}, 1.0\right)$$

---

## B. How Bearing Rate $\dot{\phi}$ is Calculated

- **Source**: [`navigation/geometry.py`](file:///d:/orca/navigation/geometry.py#L57-L74), [`navigation/geometry.py`](file:///d:/orca/navigation/geometry.py#L137-L150)
- **Numerical Differentiation**:
  Between successive observations at $t$ and $t - \Delta t$:
  $$\Delta \phi = \text{wrap}_{[-\pi, \pi]}(\phi_{\text{curr}} - \phi_{\text{prev}})$$
  $$\dot{\phi} = \frac{\Delta \phi}{\Delta t} \quad (\text{rad/s})$$
- **Analytical Alternative**: In [`navigation/threat_model.py`](file:///d:/orca/navigation/threat_model.py#L75-L82), $\dot{\phi}$ is also computed analytically from state vectors $(x, y)$ and relative ground velocities $(v_x, v_y)$:
  $$\dot{\phi} = \frac{x v_y - y v_x}{x^2 + y^2}$$

---

## C. How Tracking Quality $q_{BR}$ is Calculated

- **Source**: [`navigation/geometry.py`](file:///d:/orca/navigation/geometry.py#L76-L90)
- **Mathematical Definition**:
  $$q_{BR}(\dot{\phi}) = \exp\left(-\frac{\dot{\phi}^2}{2\sigma_{BR}^2}\right)$$
  - $\sigma_{BR}$: Angular slew rate Gaussian parameter (default: $0.60\text{ rad/s} \approx 34.4^\circ/\text{s}$).
  - Clamped to $[q_{\text{min\_quality}}, 1.0]$ with $q_{\text{min\_quality}} = 0.05$.
- **Interpretation**: When the target is moving slowly across the camera field ($\dot{\phi} \approx 0$), $q_{BR} \approx 1.0$. When the target undergoes rapid angular acceleration (e.g. crossing pedestrians or rapid head saccades), tracking stability degrades exponentially.

---

## D. Whether $q_{BR}$ Actually Enters $\rho_{FOV}$

- **Status**: **YES, directly integrated**.
- **Source**: [`navigation/rho_fov.py`](file:///d:/orca/navigation/rho_fov.py#L138-L147)
- **Audit Findings**:
  - In earlier prototype iterations, $q_{BR}$ was thresholded via a knife-edge step function $\mathbb{I}(q_{BR} \ge q_{\text{min}})$, which effectively discarded the continuous soft quality signal.
  - In the current implementation, when `use_soft_quality=True` (the default), $q_{BR}$ actively weights each trajectory:
    $$V_{\text{joint}}(s, i) = \mathbb{I}(|\phi_i(s)| \le \theta_c) \cdot \exp\left(-\frac{\dot{\phi}_{i}(s)^2}{2\sigma_{BR}^2}\right)$$
  - This continuous observation quality term directly reduces trajectory survival weight $w_i$ even when an entity remains strictly inside the horizontal FOV boundary.

---

## E. Exact Definition of Instantaneous Validity $V$

- **Source**: [`navigation/rho_fov.py`](file:///d:/orca/navigation/rho_fov.py#L75-L88)
- **Instantaneous Point Validity**:
  $$V_{c, e}(t) = \begin{cases} 1 & \text{if } |\phi_{c, e}(t)| \le \theta_c \;\land\; q_{BR}(\dot{\phi}_{c, e}(t)) \ge q_{\text{min}} \\ 0 & \text{otherwise} \end{cases}$$
- **Continuous Trajectory Validity Formulation**:
  Over sample trajectory $i$ at time step $\tau$:
  $$V_{c, e}^{(i)}(t + \tau) = \mathbb{I}(|\phi_i(\tau)| \le \theta_c) \cdot q_{BR}(\dot{\phi}_i(\tau)) \in [0, 1]$$

---

## F. Exact Monte Carlo Propagation Model

- **Source**: [`navigation/rho_fov.py`](file:///d:/orca/navigation/rho_fov.py#L122-L137)
- **State Perturbation**:
  For an entity with initial state $(\phi_0, \dot{\phi}_0)$, $N$ stochastic perturbation samples are drawn:
  $$\phi_0^{(i)} \sim \mathcal{N}(\phi_0, \sigma_\phi^2), \quad \sigma_\phi = 0.05\text{ rad}$$
  $$\dot{\phi}_0^{(i)} \sim \mathcal{N}(\dot{\phi}_0, \sigma_{\dot{\phi}}^2), \quad \sigma_{\dot{\phi}} = 0.15\text{ rad/s}$$
- **Kinematic Forward Propagation**:
  For discrete time steps $\tau \in \{\Delta t, 2\Delta t, \dots, H\}$ where $\Delta t = 0.20\text{s}$ and $H = 2.0\text{s}$ ($S = 10$ steps):
  $$\phi^{(i)}(\tau) = \phi_0^{(i)} + \dot{\phi}_0^{(i)} \cdot \tau$$
  $$\dot{\phi}^{(i)}(\tau) = \dot{\phi}_0^{(i)}$$
- **Assumptions**: Constant bearing-rate kinematics with stochastic initial state dispersion.

---

## G. Exact Formula for $\rho_{FOV}$

- **Source**: [`navigation/rho_fov.py`](file:///d:/orca/navigation/rho_fov.py#L143-L153)
- **Continuous Formulation**:
  For each perturbation trajectory $i \in \{1, \dots, N\}$:
  $$w_i = \prod_{s=1}^S \left[ \mathbb{I}(|\phi^{(i)}(s\Delta t)| \le \theta_c) \cdot q_{BR}(\dot{\phi}^{(i)}(s\Delta t)) \right]$$
  $$\rho_{FOV}(c, e) = \frac{1}{N} \sum_{i=1}^N w_i \in [0, 1]$$
- **Binary Thresholding Mode (`use_soft_quality=False`)**:
  $$w_i = \prod_{s=1}^S \left[ \mathbb{I}(|\phi^{(i)}(s\Delta t)| \le \theta_c) \cdot \mathbb{I}(q_{BR} \ge q_{\text{min}}) \right]$$
  $$\rho_{FOV}(c, e) = \frac{1}{N} \sum_{i=1}^N w_i$$

---

## H. Exact Support Calculation

- **Source**: [`navigation/camera_support.py`](file:///d:/orca/navigation/camera_support.py#L87-L102)
- **Critical Entity Set Admission $E_k$**:
  Extracted by [`navigation/critical_region.py`](file:///d:/orca/navigation/critical_region.py#L95-L100):
  $$E_k = \{e \in \mathcal{E} \mid \text{ThreatLevel}(e, G_k) \in \{\text{CRITICAL}, \text{MEDIUM}\} \;\lor\; \text{CorridorOverlap}(e, G_k) > 0.15\text{m}\}$$
- **Support Formula**:
  $$\text{Support}(c, k) = \begin{cases} 1.0 & \text{if } E_k = \emptyset \\ \min_{e \in E_k} \rho_{FOV}(c, e) & \text{if } E_k \neq \emptyset \end{cases}$$
- **Special Case**: For $G_{\text{STOP}}$, $\text{Support}(c, \text{STOP}) = 1.0$ intrinsically.

---

## I. Exact Admissibility Logic

- **Source**: [`navigation/camera_support.py`](file:///d:/orca/navigation/camera_support.py#L103-L119), [`navigation/decision.py`](file:///d:/orca/navigation/decision.py#L308-L325)
- **Corridor Admissibility Condition**:
  $$\text{Admissible}(G_k) \iff \text{SafetyScore}(G_k) \ge \tau_{\text{safe}} \;\land\; \text{Support}(c_{\text{pri}}, k) \ge \tau_{\text{cam}}$$
- **Decision Selection Hierarchy**:
  1. Directional candidates are sorted by path score: $G_{(1)}, G_{(2)}, \dots, G_{(m)}$.
  2. Candidates failing the gate with `INADMISSIBLE_LOW_SUPPORT` are excluded from `admissible_directional`.
  3. The highest-scoring admissible candidate is selected:
     $$k^* = \arg\max_{k : \text{Admissible}(G_k)} \text{Score}(G_k)$$
  4. If all directional candidates are rejected due to low perceptual support (`not admissible_directional and low_support_rejected`), the system commands an emergency protective stop with reason `LOW_PERCEPTUAL_SUPPORT`.

---

## J. Where $\tau_{\text{safe}}$ and $\tau_{\text{cam}}$ Originate

- **Source**: [`config.yaml`](file:///d:/orca/config.yaml#L216-L221), [`navigation/camera_support.py`](file:///d:/orca/navigation/camera_support.py#L51-L56), [`main.py`](file:///d:/orca/main.py#L254-L258)
- **Code Locations**:
  - `config.yaml`:
    ```yaml
    safety:
      tau_safe: 0.35
      tau_cam: 0.40
    ```
  - `PerceptualSupportGate.__init__(tau_safe=0.35, tau_cam=0.40)`
- **Physical Rationale**:
  - $\tau_{\text{safe}} = 0.35$: Corresponds to minimum lateral clearance ($< 5\text{cm}$ shoulder buffer in $0.80\text{m}$ corridor).
  - $\tau_{\text{cam}} = 0.40$: Corresponds to the $1/e \approx 0.368$ critical survival boundary for track covariance convergence over $S=10$ update steps.

---

## K. Whether Thresholds are Empirical, Heuristic, or Configurable

- **Configurable**: **YES**. Both $\tau_{\text{safe}}$ and $\tau_{\text{cam}}$ are fully configurable in `config.yaml` and parameterizable at runtime.
- **Derivation Status**: **Operational Heuristics with Physical Grounding**.
  - While physically grounded (shoulder width and Kalman covariance convergence), they have **not yet been optimized against an empirical ROC operating curve** across diverse real-world benchmarks.
  - They should be treated as **tunable operational hyperparameters**, pending the sensitivity and calibration studies in Tasks 7 and 9.

---

## L. Randomness and Seeding Behavior

- **Current State**:
  In [`navigation/rho_fov.py`](file:///d:/orca/navigation/rho_fov.py#L124-L125), perturbation samples were generated using `np.random.normal(...)` without an explicit `np.random.Generator` instance or deterministic trial seed.
- **Audit Flag**:
  Relying on ambient global NumPy random state introduces run-to-run stochastic variance across multi-threaded operations.
- **Required Action**:
  Introduce explicit, deterministic trial seeding (`seed_everything(seed)` and instance-level `np.random.default_rng(seed)`) as required by Task 2.
