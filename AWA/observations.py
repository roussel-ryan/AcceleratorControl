import numpy as np
import copy
import time
import h5py
import sys, os
import pandas as pd
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))


from accelerator_control import observations

class AWAScreen(observations.GroupObservation):
    def __init__(self, image_directory = None, n_samples = 1):
        outputs = ['FWHMX',
                   'FWHMY',
                   'FWHML',
                   'CX','CY']

        self.n_samples = 1

        #determine if we are saving images
        self.image_directory = image_directory
        if self.image_directory == None:
            self.save_image = False

        else:
            self.save_image = True

            
        super().__init__('AWAScreen', outputs)


    def __call__(self, controller):
        '''
        do screen measurement

        '''
        #wait for any previous changes to settle
        time.sleep(3)

        data_pkt = controller.interface.GetNewImage(self.n_samples)

        scalar_data = np.hstack(data_pkt[1:])
        data =  pd.DataFrame(data = scalar_data,
                             columns = self.output_names)

        #save image(s) in h5 format
        ctime = int(time.time())
        if self.save_image:
           for i in range(self.n_samples):
               with h5py.File(f'{self.image_directory}/img_{ctime}_{i}.h5','w') as f:

                   dset = f.create_dataset('raw', data = data_pkt[0][i])

                   #add attrs
                   for name, item in zip(self.output_names, data_pkt[1:]):
                       print(f'{name}:{item}')
                       
                       dset.attrs[name] = item[i]


        return data
