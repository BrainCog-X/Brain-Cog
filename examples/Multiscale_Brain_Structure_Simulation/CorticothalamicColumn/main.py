import sys
sys.path.append("../")
import time
from model import cortex_thalamus

if __name__ == '__main__':
    starttime = time.time()
    myCortex = cortex_thalamus.Cortex_Thalamus(1000)  # create a cortex object and specify the neuron number scale
    myCortex.CreateCortexNetwork()  # create cortex-thalamus network by the cortical object
    myCortex.run()
    print(len(myCortex.synapse))

    totaltime = (time.time() - starttime)
    print("totaltime:" + str(totaltime) + "s")