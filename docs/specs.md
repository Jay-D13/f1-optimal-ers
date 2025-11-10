# F1 Technical Specifications: Sources and Documentation

## Overview
This document provides detailed sourcing and technical references for the configurations used in the F1 ERS Optimal Control simulation model. All values are based on official FIA regulations and published technical data from the hybrid era (2014-2025).

---

## 1. ERSConfig Parameters

### 1.1 MGU-K Power Limits

**Current Regulations (2014-2025):**
- **Maximum Deployment Power: 120 kW (161 HP)**
- **Maximum Recovery Power: 120 kW**

**Sources:**
- FIA Formula 1 Technical Regulations
- The MGU-K is limited by regulations to a maximum power output of 120 kW, which equates to approximately 161 bhp [Source: Motorsport.com, 2023]
- Regulations specify a maximum output of 120 kW and maximum rotational speed of 50,000 rpm for the MGU-K [Source: Honda Technology, Evolution of Hybrid Technologies]
- The maximum power the MGU-K can produce is 120kW – which equates to about 160bhp [Source: Motorsport.com Insider's Guide, 2023]

**Regulatory Context:**
- The power limit is measured at the input to the MGU-K controller at 126 kW, accounting for 95% efficiency [Source: F1technical.net]
- The MGU-K is restricted at the start of races and cannot be used until the car reaches 100 km/h (current regulations) [Source: FIA Technical Regulations]

**2026 Changes:**
- The MGU-K power will increase dramatically to **350 kW** for the 2026 season
- The deployment formula for 2026: P(kW) = 1850 - [5 × car speed (km/h)] when car speed is below 340 km/h
- MGU-K recovery will increase from 2 MJ per lap to **9 MJ per lap** [Source: Autosport Forums, 2022; Formula1.com, 2024]

---

### 1.2 Battery Energy Storage

**Specified Value: 4 MJ per lap**

**Sources:**
- The battery can deploy **4 megajoules (MJ) per lap** to the MGU-K from the Energy Store (ES) [Source: The Race, 2021]
- The MGU-K can recover up to **2 MJ per lap** from braking into the ES [Source: Multiple sources]
- The actual battery capacity is **not limited** by regulations; only the state of charge (SOC) delta is regulated [Source: F1technical.net, 2018]

**Technical Details:**
- **Regulation**: "The difference between the maximum and minimum state of charge of the ES may not exceed 4 MJ at any time the car is on track" [Source: 2022 FIA Technical Regulations, Article 5.3]
- Teams can have batteries with greater capacity (e.g., 6 MJ total), as long as the swing between max and min SOC doesn't exceed 4 MJ
- Typical actual capacity: **4.5-5 MJ** to account for degradation over time [Source: The Race, 2021]
- Maximum theoretical capacity with current weight limits: approximately **19-25 MJ** based on lithium-ion energy density [Source: F1technical.net]

**Battery Physical Specifications:**
- Minimum weight: **20 kg** for battery cells and connections
- Maximum weight: **25 kg** [Source: FIA Technical Regulations]
- Minimum volume: **22 L** (increased to 22 L in 2018)
- Operating voltage: Maximum **1,000 V** [Source: Honda Technology]
- Technology: Lithium-ion cells
- Must comply with UN38.3 standards for transport [Source: The Race, 2021]

---

### 1.3 Efficiency Values

**Deployment Efficiency: 95%**
**Recovery Efficiency: 85%**

**Sources:**
- Deployment efficiency (motor mode): The MGU-K controller allows an input power of 126 kW expecting 95% motor/controller efficiency to achieve 120 kW output [Source: F1technical.net, 2019]
- Recovery efficiency is typically **80-90%** for regenerative braking systems, accounting for:
  - Motor/generator conversion losses
  - Battery charging losses
  - Power electronics losses
  - Thermal losses

**Engineering Basis:**
- Modern electric motors can achieve 95-97% efficiency in motor mode [Source: F1technical.net discussion]
- Battery round-trip efficiency for lithium-ion: typically 85-95%
- Controller/inverter efficiency: approximately 95%
- Combined system efficiency for recovery is lower due to multiple conversion stages

---

### 1.4 SOC Limits

**Minimum SOC: 0.0 (0%)**
**Maximum SOC: 1.0 (100%)**

**Regulatory Note:**
These represent the working range limits. In practice, teams may limit the operational window to preserve battery life (e.g., operating between 20-85% of physical capacity).

---

## 2. VehicleConfig Parameters

### 2.1 Vehicle Mass

**Specified Value: 798 kg** (including driver)

**Current Regulations (2023-2025):**
- Minimum weight: **798 kg** without fuel, with driver, fitted with dry-weather tires
- This was increased from the original 2022 target of 795 kg to 798 kg due to:
  - Teams struggling to meet the lower limit
  - Addition of floor stays to combat porpoising (+2 kg)
  - Mandated FIA sensors (+1 kg)
  - Heavier Pirelli tire construction
- Driver weight minimum: **80 kg** (including helmet, race suit, and shoes) [Source: Motorsport.com, 2023; GPFans, 2024]

**Historical Context:**
- 2014-2021 (start of hybrid era): 691 kg → gradually increased
- 2022 (new regulations): 795 kg → 798 kg
- 2025: **800 kg** (allowing drivers up to 82 kg)
- 2026: Target reduction to **768 kg** (30 kg lighter) [Source: GPFans, 2024; Total Motorsport, 2025]

**Component Weights:**
- Power unit (entire assembly): Minimum **150 kg** (increased to 151 kg in 2022)
  - Includes ICE, turbocharger, MGU-K, MGU-H, ES, control electronics
- Energy Store: 20-25 kg
- MGU-K: Minimum **7 kg** (increased to 16 kg for 2026)
- MGU-H: Minimum **4 kg** (to be removed in 2026)
- Steering wheel: Approximately **1.3 kg** [Source: Motorsport.com, 2023]

**Sources:**
- FIA Formula 1 Technical Regulations Articles 4.1-4.3
- Motorsport.com: "How much does an F1 car weigh in 2023" (June 2023)
- Wikipedia: Formula One Regulations (2025)

---

### 2.2 Aerodynamic Parameters

#### Frontal Area
**Specified Value: 1.5 m² (typical)**

**Sources:**
- Modern F1 cars have a frontal area of approximately **1.3 m²** to **1.5 m²**
- Red Bull F1 car (2010): Approximately **2064 square inches ≈ 1.33 m²** [Source: Road & Track, 2012]
- The frontal area has decreased over time due to regulations mandating narrower cars

**Measurement Note:**
Frontal area is the projected area when viewing the car from the front, including:
- Chassis width
- Exposed wheels
- Front wing (within track width)
- Driver's helmet

---

#### Drag Coefficient (Cd)
**Specified Value: 0.9 (typical for medium downforce configuration)**

**Sources:**
- F1 cars have drag coefficients between **0.7 and 1.0** depending on configuration [Source: Formula1-Dictionary.net]
- Monaco (high downforce): Cd ≈ **1.0**
- Monza (low downforce): Cd ≈ **0.7** [Source: TenTenths Forum, Mike Gascoyne]
- Modern F1 cars: Cd ≈ **0.85** with CdA ≈ **1.2 m²** [Source: Formula1-Dictionary.net]
- 2010 Red Bull calculation: Cd ≈ **0.98** [Source: Road & Track, 2012]

**Context:**
- F1 drag coefficients are **2-4 times higher** than modern road cars (Cd ≈ 0.25-0.3)
- High drag is a trade-off for massive downforce generation
- Exposed wheels alone account for approximately **35% of total drag** [Source: F1technical.net]
- Typical lift-to-drag ratio (Cl/Cd): **2.5-3.0** [Source: Multiple technical sources]

**Comparison:**
- Modern saloon car: Cd ≈ 0.30
- SUV: Cd ≈ 0.35-0.45
- LMP1 prototype: Lower Cd than F1 due to closed wheels
- F1 (1983, without sidepods): Cd ≈ 1.07 [Source: Race Car Aerodynamics by Joseph Katz]

---

#### Rolling Resistance Coefficient (Cr)
**Specified Value: 0.02**

**Source:**
- Standard value for racing slick tires on smooth surfaces
- Racing slicks typically have Cr between **0.01-0.025**
- Value varies with:
  - Tire compound (soft/medium/hard)
  - Tire temperature
  - Track surface condition
  - Tire pressure

**Technical Basis:**
- Much lower than road car tires (Cr ≈ 0.01-0.015 for performance road tires)
- Pirelli F1 slicks are optimized for grip rather than rolling resistance
- Value is significantly affected by tire deformation under load

---

### 2.3 Powertrain Specifications

#### Maximum ICE Power
**Specified Value: 600 kW (~800 HP)**

**Current Reality (2023-2025):**
- ICE output: Approximately **550-600 kW** (740-800 HP) at 15,000 RPM
- MGU-K contribution: **120 kW** (161 HP)
- MGU-H contribution: Variable, effectively unlimited per regulations
- **Total system output: ~750 kW (1,000+ HP)** [Source: Multiple sources]

**Sources:**
- "The ICE refers to the 1.6 litre V6 that develops around **700 horsepower** all by itself" [Source: Planet F1, 2022]
- Current total power approximately **750 kW (1000 HP)** when MGU-K is deployed [Source: FIA 2026 Regulations Document]
- ICE power approximately **630 kW (850 HP)** mentioned for current engines [Source: FIA]
- Modern F1 engines produce **830-850 HP** from the ICE alone [Source: The Manual, 2025]

**Engine Specifications:**
- Configuration: **1.6L V6** turbocharged
- Maximum RPM: **15,000 RPM** (down from 20,000+ in V8 era)
- Fuel flow limit: **100 kg/h** (regulated by mass flow)
- Compression ratio: Maximum **18:1** (geometric) [Source: Honda Technology]
- Thermal efficiency: Over **50%** (world-leading) [Source: Multiple sources]
- Number of cylinders: **6** in 90-degree V configuration

**2026 Changes:**
- ICE power will drop to approximately **400 kW** (536 HP)
- Fuel flow will be regulated by energy (MJ/h) rather than mass
- Target thermal efficiency: **48%**
- MGU-K power will increase to **350 kW** to compensate [Source: Autosport Forums, 2022]

---

#### Maximum Brake Force
**Specified Value: 50 kN (50,000 N)**

**Engineering Basis:**
- Modern F1 carbon-carbon brakes can generate enormous forces
- Brake pedal force: 100-120 kg typical, up to 180 kg maximum
- Brake system hydraulic pressure: 120-150 bar typical
- Front brake disc diameter: Up to 370 mm (2022+ regulations: 330 mm max)
- Rear brake disc diameter: Up to 370 mm (2022+ regulations: 370 mm max)

**Performance Context:**
- F1 cars can brake from 200 km/h to 0 in approximately **2-2.5 seconds**
- Peak braking deceleration: **5-6 g**
- Braking distance from 100 km/h: Approximately **17 meters**
- With MGU-K recovery, drivers can achieve **1 g deceleration** without touching brakes at high speed [Source: ScienceDirect]

**2026 Note:**
- Rear mechanical brakes may be nearly eliminated due to increased MGU-K recovery
- Similar to Formula E Gen3 cars (no rear mechanical brakes) [Source: TheDrive, 2022]

---

## 3. Vehicle Dynamics Model Sources

### 3.1 Longitudinal Dynamics Equations

The simplified point-mass model is based on standard vehicle dynamics theory:

**Drag Force:**
```
F_drag = 0.5 × ρ × Cd × A × v²
```
Where:
- ρ = air density (1.225 kg/m³ at sea level)
- Cd = drag coefficient
- A = frontal area
- v = velocity

**Source:** Standard fluid dynamics, found in:
- Gillespie, T.D. "Fundamentals of Vehicle Dynamics" (SAE International, 1992)
- Milliken & Milliken "Race Car Vehicle Dynamics" (SAE International, 1995)

---

**Rolling Resistance:**
```
F_rolling = m × g × Cr × cos(θ)
```
Where:
- m = vehicle mass
- g = gravitational acceleration (9.81 m/s²)
- Cr = rolling resistance coefficient
- θ = road gradient

**Source:** Standard vehicle dynamics, accounting for normal force variation with gradient

---

**Gravitational Force:**
```
F_gravity = m × g × sin(θ)
```

**Source:** Basic physics, component of weight along slope

---

**Traction Force:**
```
F_traction = P_total / v
```
Where:
- P_total = ICE power + ERS deployment power
- v = velocity

**Limitations:**
This is a simplified model that doesn't account for:
- Tire slip and traction limits
- Downforce effects on normal force
- Detailed powertrain dynamics
- Gear ratios

**More accurate models would use:**
- Pacejka tire models
- Longitudinal slip dynamics
- Downforce: F_down = 0.5 × ρ × Cl × A × v²
- Effective normal force including downforce for traction limits

---

### 3.2 Battery Dynamics

**Energy Flow:**
```
P_battery = -P_ERS / η_deployment     (when deploying, P_ERS > 0)
P_battery = -P_ERS × η_recovery       (when recovering, P_ERS < 0)
```

**State of Charge Rate:**
```
dSOC/dt = P_battery / E_capacity
```

**Source:**
- Standard battery modeling approach
- Assumes constant efficiency (simplification)
- Real systems have efficiency curves dependent on:
  - State of charge
  - Power level
  - Temperature
  - Battery age

---

## 4. Model Limitations and Simplifications

### What's Missing from Your Model:

1. **Lateral Dynamics**
   - No cornering forces
   - No slip angles
   - No vehicle balance modeling

2. **Aerodynamics**
   - No downforce modeling
   - Constant drag coefficient (reality: varies with ride height, wing angles)
   - No DRS effects

3. **Tire Model**
   - No grip limits
   - No tire temperature
   - No compound degradation
   - Constant rolling resistance

4. **Detailed Powertrain**
   - No gear ratios
   - Simplified power delivery
   - No turbo lag
   - No MGU-H modeling (correct for 2026+)

5. **Thermal Effects**
   - No brake temperature
   - No tire temperature
   - No PU temperature limits

6. **Energy Management Details**
   - No SOC-dependent efficiency
   - No energy deployment limitations beyond simple power limits

---

## 5. References and Further Reading

### Official Sources:
1. FIA Formula 1 Technical Regulations (2023, 2024, 2025, 2026)
2. FIA Formula 1 Sporting Regulations

### Technical Publications:
1. Gillespie, T.D. (1992). "Fundamentals of Vehicle Dynamics." SAE International
2. Milliken, W.F. & Milliken, D.L. (1995). "Race Car Vehicle Dynamics." SAE International
3. Katz, J. (1995). "Race Car Aerodynamics." Bentley Publishers
4. Dominy, R.G. (1992). "Aerodynamics of Grand Prix Cars." Proceedings of the Institution of Mechanical Engineers

### Online Resources:
1. Formula1.com - Official F1 website
2. FIA.com - Governing body regulations
3. Motorsport.com - Technical analysis
4. F1technical.net - Technical forums and discussions
5. The Race - Technical journalism
6. Honda F1 Technology - Manufacturer insights

### Academic Papers:
1. Perantoni, G. & Limebeer, D.J.N. (2014). "Optimal control for a Formula One car with variable parameters"
2. Casanova, D. (2000). "On Minimum Time Vehicle Manoeuvring"
3. Kelly, D.P. (2008). "Lap Time Simulation with Transient Vehicle and Tyre Dynamics"

---

## 6. Summary Table

| Parameter | Value | Source Type | Accuracy |
|-----------|-------|-------------|----------|
| MGU-K Power | 120 kW | FIA Regulations | Exact |
| ES per lap | 4 MJ | FIA Regulations | Exact |
| Battery SOC range | 0-4 MJ | FIA Regulations | Exact |
| Vehicle mass | 798 kg | FIA Regulations | Exact |
| ICE power | ~600 kW | Industry estimates | ±50 kW |
| Cd | 0.7-1.0 | Published data | ±0.1 |
| Frontal area | ~1.4 m² | Technical analysis | ±0.2 m² |
| Cr | 0.02 | Engineering estimate | ±0.005 |
| Deployment efficiency | 95% | Technical analysis | ±2% |
| Recovery efficiency | 85% | Engineering estimate | ±5% |

---

## Document Information

**Last Updated:** November 10, 2025
**Author:** Technical Documentation
**Purpose:** Source verification for F1 ERS simulation parameters
**Status:** Current for 2023-2025 regulations

**Note:** 2026 regulations will bring significant changes, particularly:
- MGU-K power: 120 kW → 350 kW
- ICE power: ~600 kW → ~400 kW
- Vehicle mass target: 798 kg → 768 kg
- MGU-H removal
- Sustainable fuels mandatory