from IPython.display import clear_output
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

if __name__ == '__main__':
    pass

for i in range(1000):
    clear_output(wait=True)
    data_dict = {}
    data_dict['category'] = ['category 1','category 2','category 3']
    data_dict['lower'] = [0.1,0.2,0.15]
    data_dict['upper'] = [0.22,np.random.uniform(0.2,0.3),0.21]
    dataset = pd.DataFrame(data_dict)
    for lower,upper,y in zip(dataset['lower'],dataset['upper'],range(len(dataset))):
        plt.plot((lower,upper),(y,y),'ro-',color='orange')
    plt.yticks(range(len(dataset)),list(dataset['category']));
    plt.show()
    sleep(1)