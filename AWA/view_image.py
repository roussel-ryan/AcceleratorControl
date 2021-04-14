import h5py
import matplotlib.pyplot as plt

for i in [25, 40, 48]:
    base_fname = 'pics/5nC_3_param_sigma_100_2/img_'
    fname = base_fname + f'{i}.h5'

    with h5py.File(fname) as f:
        image = f['image_1']
        fig, ax = plt.subplots()
        ax.imshow(image)

        ax.set_title(dict(image.attrs))

plt.show()
