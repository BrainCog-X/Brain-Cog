## Input:

* Program to enter a table of connection weights between 213 brain regions, which is saved in 'mouse_weight.pt'. The brain regions' name and neuron number are saved in 'mouse_brain_region.xlsx'. These two files are available in the follow link:

https://drive.google.com/drive/folders/1MWHY52gKPGneBEJxJN9DzE7thnLrhG1j?usp=sharing

## output

* The program generates a data file of the individual neuron firing time points recorded, and the large number of data points requires the use of drawing software to display the results.

## setting:

* scale: The scale of the number of neurons
* neuron_model: ‘HHNode’ or ‘aEIF’
* weight_matrix: Matrix of the number of synaptic connections between brain regions
* neuron_num: The number of neurons in each brain region
* ratio: the ratio of each neuron type in each brain region
* syn_num: average number of synapses per neuron within region 
