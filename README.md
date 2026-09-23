# Opinion Dynamics and Echo Chambers

This repository contains the Python implementation developed for the simulation of opinion dynamics, information diffusion, and echo chamber formation in social networks.

The model represents users as nodes in a directed social network. Each user has a continuous opinion, and the same tolerance level is  assumed for all users . Publications propagate through the network, influencing the opinions of users who are exposed to and accept the content. Network connections can also evolve according to ideological distance and social recommendation.

## Model components

The computational model consists of the following components:

* Social network generation using a Barabási-Albert network and Erdos Renyi.
* Construction of a directed follower network.
* Continuous initial opinions in the interval \([-1,1]\).
* User tolerance.
* Information diffusion through the network.
* Opinion updating after publication exposure.
* Network disconnections based on opinion distance.
* Formation of new connections through social recommendation and opinions compatibility.
* Calculation of the \(BC_{hom}\) bimodality coefficient.
* Identification of groups of users with similar opinions.

## Main parameters

| Parameter             |           Value |
| --------------------- | --------------: |
| Number of users       |            1000 |
| Number of periods     |              20 |
| Tolerance             |             0.5 |
| Initial opinion range |         [-1, 1] |
| Network model         | Barabási-Albert or Erdős–Rényi |
| \(m\)                 |               3 |
| Network seed          |              42 |
| Opinion seed          |             123 |

Two publications are considered in the main simulation. However, you can change the values of $\sigma$ and other parameters in the algorithm.
$$
\sigma_1 =0.3615758 
$$

$$
\sigma_2 = 0.673157
$$
The values you can be changed in the model

$
\sigma_3=-0.6170859
$

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Running the model

Run the main simulation with:

```bash
python opinion_dynamics_model.py
```

The model generates figures describing the evolution of opinions, information diffusion, network connections, and the \(BC_{hom}\) measure.

## Repository structure

```text
opinion-dynamics-echo-chambers/
│
├── README.md
├── opinion_dynamics_model.py
├── Dinamics.py
├── requirements.txt
├── .gitignore
│
└── figures/
    ├── BChom.png
    ├── conexiones_desconexiones.png
    ├── informados.png
    ├── grafica_opiniones.png
    ├── grafica_hist.png
    ├── bNNvsS.png
    └── bNNvsSfinal.png
```

## Results

### Evolution of opinions

![Opinion dynamics](figures/grafica_opiniones.png)

### Information diffusion

![Information diffusion](figures/informados.png)

### Network connections and disconnections

![Connections and disconnections](figures/conexiones_desconexiones.png)

### \(BC_{hom}\)

![BC hom](figures/BChom.png)

### Opinion and neighborhood opinion

![Initial state](figures/bNNvsS.png)

![Final state](figures/bNNvsSfinal.png)

![Histograma](figures/grafica_hist.png)

## Research context

This computational model was developed as part of a master's thesis in Mathematical Economics focused on opinion dynamics and echo chamber formation in social networks.

The model studies the interaction between opinion dynamics, information diffusion, network structure, adaptive connections, and the formation of groups with similar opinions.

## Reproducibility

Random seeds are specified for the generation of the social network and the initial opinions to facilitate reproducibility of the numerical experiments.

## Author

Iván Emmanuel Granja Carranco

Master's Program in Mathematical Economics
Universidad Autónoma de San Luis Potosí
