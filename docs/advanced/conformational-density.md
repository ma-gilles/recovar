# Conformational Density

RECOVAR can estimate the probability density of conformations in the latent space. This enables:

- Free energy estimation (free energy = -kT ln(density))
- Identification of stable conformational states
- High-density trajectory computation

## Estimating density

```bash
recovar estimate_conformational_density output --zdim=10 \
    -o density_output
```

This produces density estimates in the latent space.

## Estimating stable states

```bash
recovar estimate_stable_states output --zdim=10 \
    -o stable_states --density density_output/deconv_density_knee.pkl
```

Identifies local minima in the free energy landscape.

## Using density for trajectories

The density can guide trajectory computation to follow low free-energy paths:

```bash
recovar compute_trajectory output -o trajectory --zdim=10 \
    --density density_output/deconv_density_knee.pkl \
    --endpts centers.txt --ind 0,1
```

Without `--density`, trajectories follow straight lines in latent space. With density, they curve to follow high-density (low free-energy) regions.

## Interpretation

The conformational density represents how many particles occupy each region of the latent space. Under equilibrium assumptions:

$$\Delta G = -k_B T \ln\left(\frac{\rho}{\rho_0}\right)$$

where $\rho$ is the conformational density and $\rho_0$ is a reference density.

!!! note
    The relationship between conformational density and free energy assumes the dataset represents an equilibrium ensemble. This may not always be the case (e.g., with preferential orientation or during active processes).
