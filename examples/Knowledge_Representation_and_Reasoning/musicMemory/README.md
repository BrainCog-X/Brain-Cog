usic Memory and stylistic composition

This repository contains code from our paper:
- [**Temporal-Sequential Learning With a Brain-Inspired Spiking Neural Network and Its Application to Musical Memory**](https://www.cell.com/patterns/fulltext/S2666-3899(22)00119-2) published in Frontiers in Computational Neuroscience. **https://www.cell.com/patterns/fulltext/S2666-3899(22)00119-2**,
- [**Stylistic composition of melodies based on a brain-inspired spiking neural network**](https://www.frontiersin.org/articles/10.3389/fnsys.2021.639484/full) published in  Frontiers in Systems Neuroscience **https://www.frontiersin.org/articles/10.3389/fnsys.2021.639484/full**.

If you use our code or refer to this project, please cite this paper.

## Requirments

* numpy
* scipy
* pytorch >= 1.7.0
* pretty_midi >= 0.2.9


## Data preparation

The dataset used here can be referred to the website http://www.piano-midi.de/. 


## Run
* Run the script *task/musicMemory.py* to memorize and recall the musical melodies, the result will be recorded in a midi file.
* Run the script *task/musicGeneration.py* to learn and generate melodies with different styles, the result will be recorded in a midi file.
The api and details can be found in these scripts. 

