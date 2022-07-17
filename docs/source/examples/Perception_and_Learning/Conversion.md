# Conversion Method
Training deep spiking neural network with ann-snn conversion
replace ReLU and MaxPooling in pytorch model to make origin ANN to be converted SNN to finish complex tasks

## Results
```shell
python CIFAR10_VGG16.py
python converted_CIFAR10.py
```

You should first run the `CIFAR10_VGG16.py` to get a well-trained ANN.
Then `converted_CIFAR10.py` can be used to run the snn inference process.