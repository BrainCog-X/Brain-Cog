# Multisensory Integration

In `MultisensoryIntegrationDEMO_AM.py` and `MultisensoryIntegrationDEMO_AM.py`, we implement the SNNs based multisensory integration framework. To load the dataset, preprocess it and get the weights with the function `get_concept_datase_dic_and_initial_weights_lst()`​. We use `IMNet`​ or ​`AMNet`​ to describe the structure of the IM/AM model. For presynaptic neuron, we use the function `convert_vec_into_spike_trains()​` to generate the spike trains.

While for postsynaptic neuron, we use the function `reducing_tol_binarycode()​` to get the multisensory integrated output for each concept. And  ​*tol*​ is the only parameter.

In `measure_and_visualization.py​`, we will measure and visualize the results.

## Multisensory Dataset 

When implement the model in braincog, we use the famous multisensory dataset--BBSR.

Some examples are as follows:

| Concept   | Visual      | Somatic   | Audiation   | Taste    | Smell    |
| --------- | ----------- | --------- | ----------- | -------- | -------- |
| advantage | 0.213333333 | 0.032     | 0           | 0        | 0        |
| arm       | 2.5111112   | 2.2733334 | 0.133333286 | 0.233333 | 0.4      |
| ball      | 1.9580246   | 2.3111112 | 0.523809429 | 0.185185 | 0.111111 |
| baseball  | 2.2714286   | 2.6071428 | 0.352040714 | 0.071429 | 0.392857 |
| bee       | 2.795698933 | 2.4129034 | 2.096774286 | 0.290323 | 0.419355 |
| beer      | 1.4866666   | 2.2533334 | 0.190476286 | 5.8      | 4.6      |
| bird      | 2.7632184   | 2.027586  | 3.064039286 | 1.068966 | 0.517241 |
| car       | 2.521839133 | 2.9517244 | 2.216748857 | 0        | 2.206897 |
| foot      | 2.664444533 | 2.58      | 0.380952429 | 0.433333 | 3        |
| honey     | 1.757142867 | 2.3214286 | 0.015306143 | 5.642857 | 4.535714 |

## How to Run 

To get the multisensory integrated vectors:

```
cd examples/MultisensoryIntegration/code
python MultisensoryIntegrationDEMO_AM.py
python MultisensoryIntegrationDEMO_IM.py
```

To measure and analysis the vectors:

```
cd examples/MultisensoryIntegration/code
python measure_and_visualization.py
```







