# ORCA EYE — STAGE B IMPLEMENTATION AUDIT
**Two-Camera Predictive Observation Preservation: Architecture, Interfaces, & Mathematical Foundations**

---

## 1. Objective & Scope

Stage A validated that continuous observation survival $\rho_{\text{FOV}}(c, e, t, H)$ can predict when a camera $c$ will lose sight of a critical entity $e$ along a candidate navigation corridor $k$. In single-camera operation, observation loss leaves the system blind, forcing it to reject the corridor or execute a conservative stop.

**Stage B extends this capability to a two-camera system:**
$$\text{CAM0 (PRIMARY: } \text{yaw } 0^\circ, \text{HFOV } 65^\circ) \quad \text{and} \quad \text{CAM1 (PERIPHERAL: } \text{yaw } +50^\circ, \text{HFOV } 65^\circ)$$

The core research hypothesis of Stage B is:
> **When the primary camera is predicted to lose observation of a critical entity ($\rho_0 < \tau_{\text{release}}$), a secondary camera can be proactively prepared (PRE-ARM) and assigned responsibility (TRANSFER) BEFORE the actual loss occurs, preserving corridor-level perceptual support.**

---

## 2. Existing Stage-A Codebase Audit & Reuse Strategy

| Existing Module | Implementation File | Current Stage-A Role | Stage-B Reuse / Extension Strategy |
|---|---|---|---|
| **Pinhole Intrinsics & Ray Tracing** | `perception/calibration.py` | Calculates $f_x, f_y, c_x, c_y$ and half-HFOV $\theta_c = 32.5^\circ = 0.5672\text{ rad}$. | Reused directly. Two camera instances ($\text{CAM0}, \text{CAM1}$) share intrinsics and are differentiated by their extrinsic yaw angle $\psi_c$. |
| **Ground-Plane Bearing & Rate** | `navigation/geometry.py` | Computes egocentric bearing $\phi = \text{atan2}(x, y)$ and bearing rate $\dot{\phi}$. | Reused directly. Transformed into camera-local bearings $\phi_c = \phi - \psi_c$. |
| **Tracking Quality Function** | `navigation/geometry.py` | $q_{\text{BR}}(\dot{\phi}) = \exp(-\dot{\phi}^2 / (2\sigma_{\text{BR}}^2))$. | Reused directly. Soft continuous weighting is preserved without mutation. |
| **Survival Estimator ($\rho_{\text{FOV}}$)** | `navigation/rho_fov.py` | Vectorized Monte Carlo continuous survival probability over horizon $H=2.0\text{s}$. | Reused directly. Instantiated per camera: $\rho_{\text{FOV}}(\text{CAM0}, e)$ and $\rho_{\text{FOV}}(\text{CAM1}, e)$. |
| **Critical Region Extractor** | `navigation/critical_region.py` | Extracts critical entity set $E_k$ for candidate corridor $G_k$. | Reused directly. $E_k$ is defined in the wearer's ground frame, independent of camera count. |
| **Single-Camera Support Gate** | `navigation/camera_support.py` | Evaluates single-camera support $\text{Support}(c, k) = \min_{e \in E_k} \rho_{\text{FOV}}(c, e)$. | Retained for Mode B0 (Baseline single-camera mode). |
| **Responsibility State Machine** | `navigation/camera_responsibility.py` | Single-camera state machine (`PRIMARY`, `SUPPORT`, `RESERVED`, `STANDBY`). | Generalized in `navigation/camera_handover.py` to entity-specific two-camera state machine. |
| **Decision Maker** | `navigation/decision.py` | Selects admissible path candidate under $\tau_{\text{safe}}$ and $\tau_{\text{cam}}$ gates. | Reused directly. In Stage B, accepts multi-camera corridor support $\text{Support}_B(k)$. |

---

## 3. Two-Camera Geometric Configuration & Overlap Analysis

### 3.1 Extrinsic Calibration & Optical Angles
* **Camera 0 (CAM0 / PRIMARY):**
  - Mounting location: Center front, $(x_0, y_0, z_0) = (0.0, 0.0, 1.4\text{m})$
  - Yaw offset: $\psi_0 = 0.0^\circ = 0.0\text{ rad}$
  - Optical coverage: $\phi_w \in [-32.5^\circ, +32.5^\circ]$
* **Camera 1 (CAM1 / PERIPHERAL):**
  - Mounting location: Right temple / shoulder, $(x_1, y_1, z_1) \approx (+0.08\text{m}, 0.0, 1.4\text{m})$
  - Yaw offset: $\psi_1 = +50.0^\circ = +0.8727\text{ rad}$
  - Optical coverage: $\phi_w \in [+50.0^\circ - 32.5^\circ, +50.0^\circ + 32.5^\circ] = [+17.5^\circ, +82.5^\circ]$

### 3.2 Calibrated Overlap Zone
The geometric overlap between CAM0 and CAM1 occurs where both angular fields intersect:
$$\Omega_{\text{overlap}} = [\psi_1 - \theta_c, \psi_0 + \theta_c] = [+17.5^\circ, +32.5^\circ]$$
$$\text{Angular Overlap Width} = 32.5^\circ - 17.5^\circ = 15.0^\circ$$

Composite system coverage spans:
$$\Omega_{\text{total}} = [-32.5^\circ, +82.5^\circ] = 115.0^\circ \text{ (vs } 65.0^\circ \text{ in single-camera Stage A)}$$

---

## 4. Multi-Camera Entity State & Cross-Camera Association

### 4.1 Coordinate Transformation to Global Frame
Let an entity detection in camera $c$ have local bearing $\phi_c$ and estimated ground distance $d$. The egocentric world coordinates $(x_w, y_w)$ in the wearer's forward frame are:
$$\phi_w = \phi_c + \psi_c$$
$$x_w = d \cdot \sin(\phi_w), \quad y_w = d \cdot \cos(\phi_w)$$

### 4.2 Cross-Camera Association Score
When an entity enters the overlap zone $\Omega_{\text{overlap}}$, CAM0 observes track $e_0$ and CAM1 observes candidate track $e_1$. The association affinity $A(c_0, c_1, e_0, e_1)$ combines spatial, kinematic, and semantic attributes:
$$A(e_0, e_1) = w_p A_{\text{pos}}(e_0, e_1) + w_v A_{\text{vel}}(e_0, e_1) + w_c A_{\text{class}}(e_0, e_1) + w_t A_{\text{time}}(e_0, e_1)$$
where:
* $A_{\text{pos}} = \exp\left(-\frac{\|\mathbf{p}_{w,0} - \mathbf{p}_{w,1}\|^2}{2 \sigma_{\text{pos}}^2}\right)$ with hard gate: $\|\mathbf{p}_{w,0} - \mathbf{p}_{w,1}\| \le d_{\text{max}} = 1.2\text{m}$.
* $A_{\text{vel}} = \exp\left(-\frac{\|\mathbf{v}_{w,0} - \mathbf{v}_{w,1}\|^2}{2 \sigma_{\text{vel}}^2}\right)$.
* $A_{\text{class}} = 1.0$ if $\text{class}_0 = \text{class}_1$, else $0.0$.
* $A_{\text{time}} = \exp\left(-\frac{|\Delta t|}{\tau_{\text{sync}}}\right)$.

---

## 5. Predictive Handover State Machine

For each tracked entity $e$, the responsibility controller maintains state $S(e) \in \{\text{PRIMARY}, \text{PRE\_ARM}, \text{TRANSFER}, \text{SECONDARY}, \text{RELEASED}\}$.

```text
    ┌───────────┐
    │  PRIMARY  │  (CAM0 responsible)
    └─────┬─────┘
          │
          │ Trigger 1: ρ_0 < τ_release (0.40) AND ρ_1 > τ_acquire (0.60)
          ▼
    ┌───────────┐
    │  PRE_ARM  │  (CAM1 activated & tracking target in overlap)
    └─────┬─────┘
          │
          │ Trigger 2: A_01 > τ_assoc (0.70) AND dwell_time >= 0.5s
          ▼
    ┌───────────┐
    │ TRANSFER  │  (CAM1 assumes responsibility before CAM0 loss)
    └─────┬─────┘
          │
          │ Handover confirmed
          ▼
    ┌───────────┐
    │ SECONDARY │  (CAM1 responsible, CAM0 becomes RELEASED)
    └───────────┘
```

### Key Safety Property:
In a reactive system, transfer occurs only at $t_{\text{loss}}$ when $\text{CAM0}$ drops the entity, resulting in an observation gap $G = t_{\text{acquire}} - t_{\text{loss}} > 0$.
In Stage B predictive handover, transfer occurs at $t_{\text{transfer}} < t_{\text{actual loss}}$, yielding **Handover Lead Time $T_{\text{lead}} = t_{\text{actual loss}} - t_{\text{transfer}} > 0$** and **Observation Gap $G \approx 0$**.

---

## 6. Multi-Camera Corridor Support Formulation

In Stage A (single camera):
$$\text{Support}_A(k) = \min_{e \in E_k} \rho_{\text{FOV}}(\text{PRIMARY}, e)$$

In Stage B (multi-camera):
Every critical entity $e \in E_k$ can be monitored by the best available camera that has accepted responsibility or can preserve observation:
$$\rho_e^*(t, H) = \max_{c \in \{\text{CAM0}, \text{CAM1}\}} \rho_{\text{FOV}}(c, e, t, H)$$
$$\text{Support}_B(k, t) = \min_{e \in E_k} \max_{c \in \{\text{CAM0}, \text{CAM1}\}} \rho_{\text{FOV}}(c, e, t, H)$$

Corridor $k$ is admissible if and only if:
$$\text{Admissible}_B(k) \iff \text{Safety}(k) \ge \tau_{\text{safe}} \quad \land \quad \text{Support}_B(k, t) \ge \tau_{\text{cam}}$$

---

## 7. Experimental Modes for Scientific Isolation

To cleanly validate the new predictive mechanism, all navigation weights, geometry, and tracking logic are held strictly constant across three operating modes:

1. **Mode B0 (Single Camera Baseline):**
   - Uses CAM0 only.
   - Gating: $\text{Support}(k) = \min_{e \in E_k} \rho_{\text{FOV}}(\text{CAM0}, e)$.
2. **Mode B1 (Two-Camera Reactive Handover):**
   - Both CAM0 and CAM1 active.
   - CAM1 is only engaged *after* CAM0 completely loses the entity ($|\phi_0| > \theta_c$).
   - Yields observation dropout gap $G > 0$.
3. **Mode B2 (Two-Camera Predictive Handover):**
   - Full Stage-B implementation.
   - Pre-arms CAM1 when $\rho_0 < \tau_{\text{release}}$ and executes predictive transfer before loss.
   - Eliminates observation gap ($G \approx 0$) and yields positive lead time $T_{\text{lead}} > 0$.
