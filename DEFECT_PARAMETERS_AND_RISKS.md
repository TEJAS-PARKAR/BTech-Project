# Defect Parameters and Risk Scores

This guide describes the synthetic defect controls in the Streamlit dashboard and how their ground-truth risk scores are calculated. These values are generated from formulas in `data/dataset_generator.py`; they are not measurements from a physical assembly.

## Dashboard Parameters

| Parameter | Dashboard range and default | Effect on generated sample | Effect on ground-truth labels |
|---|---|---|---|
| Radial deviation | -0.15 to +0.15; default 0.0; step 0.005 | Changes the relative fit dimensions. Positive values generally make the fit tighter; negative values generally create more clearance. The exact geometric operation varies by assembly type. | Positive values increase interference risk; negative values increase clearance risk. The score saturates at a deviation magnitude of 10% of nominal radius. |
| Tilt angle | 0 to 10 degrees; default 0; step 0.1 | Rotates one mating part to represent angular misalignment. | Increases alignment risk, reaching its maximum tilt contribution at 5 degrees. |
| Lateral offset | 0 to 0.1; default 0; step 0.002 | Shifts one mating part sideways to represent eccentricity. | Increases alignment risk, reaching its maximum offset contribution at 5% of nominal radius. |
| Scanner noise sigma | 0 to 0.01; default 0.002; step 0.001 | Adds Gaussian noise to point coordinates, making the point cloud less precise. | Does not directly change ground-truth risk labels. It can make the geometry harder for a model to interpret. |

The dashboard uses a nominal radius of 1.0, so its radial-deviation and lateral-offset values are in the generator's coordinate units, not necessarily millimeters. The number-of-points control changes point-cloud sampling density; it is not a defect parameter and does not change the labels.

The dashboard also adds fixed normal-vector noise with standard deviation 0.01. This is separate from the scanner-noise slider, which controls coordinate noise.

## Risk Calculation

Let `r` be nominal radius, `d` radial deviation, `theta` tilt in degrees, and `e` lateral offset. The generator computes:

```text
interference = clip(max(0,  d / (0.1 * r)), 0, 1)
clearance    = clip(max(0, -d / (0.1 * r)), 0, 1)

T = min(1, abs(theta) / 5)
E = min(1, abs(e) / (0.05 * r))
alignment = clip(0.6 * T + 0.4 * E, 0, 1)

failure_score = clip(
    0.35 * interference
  + 0.25 * clearance
  + 0.40 * alignment,
  0, 1
)
```

Each component score is between 0 and 1. The overall failure score is converted to a class as follows:

| Failure score | Risk level |
|---|---|
| Less than 0.33 | Low (0) |
| 0.33 to less than 0.66 | Medium (1) |
| 0.66 or greater | High (2) |

The component scores describe different concerns:

- **Interference:** parts are too tight or overlap, which can make assembly difficult or cause damage.
- **Clearance:** the gap is larger than intended, which can contribute to looseness, vibration, or wear.
- **Alignment:** angular tilt and sideways offset may cause binding or uneven loading.
- **Failure score:** a weighted combination of the three scores, not a separately measured failure probability.

A single high component score does not necessarily produce a High overall class. For example, maximum interference alone contributes 0.35 to the failure score, which is Medium under these thresholds. High risk generally requires multiple substantial contributors.

## Assembly-Type Caveat

`radial_deviation` is not applied identically by all four generators:

- **Shaft and hole:** changes the shaft radius while the hole radius stays fixed.
- **Peg in hole:** changes the slot width and height in the opposite direction, so a positive value makes the fit tighter.
- **Dovetail:** changes the tongue width, so a positive value makes the fit tighter.
- **Flange joint:** shifts one flange along the Z axis. A positive value increases the gap in the current geometry, but the shared label formula counts positive deviation as increased interference. This can make the generated flange geometry disagree with its ground-truth interference label. Treat flange radial-deviation labels with caution until that generator/label mapping is corrected.

Tilt and lateral offset are applied to one part across the generators and contribute to alignment labels using the same formula.

## Interpreting Dashboard Results

When model inference is off for a generated sample, the dashboard displays the generator's formula-based labels. When model inference is on, it displays model predictions instead. For uploaded point clouds, the app has no ground-truth labels unless they are supplied separately; displayed risk values should therefore be treated as model outputs, not verified measurements.

These synthetic scores are useful for controlled experiments and training, but they are not calibrated against real failure data and do not implement a tolerance standard, contact analysis, material model, or load condition. Do not use them alone to approve a real assembly or make safety-critical decisions.
