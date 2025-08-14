# Murder_drones (WIP)

**Status: Actively in development — expect breaking changes, incomplete features, and placeholder code.**

This project aims to create an **end-to-end, permutation-invariant neural controller** for multirotor drones.  
The goal: one trained model that works on *any* drone geometry, motor type, or material — without retraining — by feeding it per-actuator "cards" describing the hardware.

## Current Goals
- [ ] Build a **simulation-first training pipeline** (MuJoCo → Isaac Lab)
- [ ] Implement **shared encoder + attention** policy for actuator cards
- [ ] Add **domain randomization** (geometry, mass, motors, wind, sensor noise)
- [ ] Model ~1 lb rigid payload in sim
- [ ] Integrate **safety wrapper** for real-world deployment
- [ ] Support **boot-time auto-calibration** for plug-and-play use
- [ ] Export to **Jetson / Pi + Coral TPU** for onboard inference

## Key Concepts
- **Actuator Cards** — Per-motor data packets:
  - Position `rᵢ`
  - Thrust axis `uᵢ`
  - Spin `sᵢ`
  - Thrust/torque coeffs `kT`, `kQ`
  - Lag `τᵢ`
  - Limits & rate constraints
- **Permutation-Invariance** — Shared encoder processes each card independently, attention layer fuses results → works regardless of motor order.
- **Domain Randomization** — Training sees endless combinations of frames, payloads, and conditions → model generalizes to new drones.

## Project Status
- **Environment**: Basic MuJoCo quad sim runs locally [working]
- **Policy**: Architecture design in progress, probably will end up using shared encoder with cross attention, maybe as differentiable (necessary for end-to-end) control allocation layer [in progress]
- **Training**: Cloud pipeline not yet tested [not started]
- **Deployment**: No hardware tests yet [not started]

## Planned Pipeline
1. **Local sanity check** — run a single fixed-geometry quad to verify reward signals.
2. **Parallel cloud training** — AWS Spot with domain randomization.
3. **Export** — Convert trained model to TensorRT/TFLite for onboard runtime.
4. **Real-world tests** — Deploy with safety wrapper on various physical drones.

## Vision
A user with no technical skill should be able to:
1. Assemble any compatible multirotor.
2. Power on — auto-calibration runs.
3. Fly — the AI adapts instantly.

## Warning
This is **research code**. Nothing here is safe for real-world flight yet.  
Always test in simulation first.

## License
TODO: Will make something ridiculous later 
