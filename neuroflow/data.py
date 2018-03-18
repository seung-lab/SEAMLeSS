import numpy as np
from cavelab.data import tfdata
import cavelab as cl
# Data preprocessing
similar = True
class Data():
    def __init__(self, hparams, random=True):
        self.batch_size = hparams.batch_size
        self.data = tfdata(hparams.train_file,
                                random_order = random,
                                batch_size = 1,#self.batch_size,
                                features=hparams.features,
                                flipping=hparams.augmentation["flipping"],
                                random_brightness=hparams.augmentation["random_brightness"],
                                random_elastic_transform=hparams.augmentation["random_elastic_transform"])
        self.similar = True
        self.test_data = np.load('data/evaluate/simple_test.npy').astype(np.float32)
        self.levels = hparams.levels

    def get_batch(self):

        xs  = self.data.get_batch() # 5,256,256,8
        start = 0
        start_level = 1
        xs = np.transpose(xs['image'].squeeze(0)[start_level:start_level+self.levels,:,:,start:start+self.batch_size], (0,3,1,2)).astype(np.float32)
        xs = Data.augmentation(xs)
        xs = np.ndarray.copy(xs)
        return xs

    ### Augmentations
    @staticmethod
    def augmentation(image): # [d,b, width, height]
        image = Data.flipping(image)
        #image = Data.translate(image)
        #TODO Elastic transform (hard)
        return image

    @staticmethod
    def flipping(xs):
        #FIXME Some of the strides are zero
        if np.random.randint(2):
            image = np.flip(xs, 2)
        if np.random.randint(2):
            image = np.flip(xs, 3)

        xs = np.rot90(xs, k=np.random.randint(4), axes=(2,3))
        return xs

    @staticmethod
    def translate(xs):
        for b in range(xs.shape[1]):
            n = np.random.randint(2*2**xs.shape[0])-2**xs.shape[0]
            m = np.random.randint(2*2**xs.shape[0])-2**xs.shape[0]

            for d in range(xs.shape[0]):
                xs[d,b] = Data.shift(xs[d,b], int(n/2**d), int(m/2**d))
        return xs

    @staticmethod
    def shift(xs, n, m):
        e = np.zeros_like(xs)
        if n > 0 and m > 0:
            e[n:,m:] = xs[:-n, :-m]
        elif n > 0 and m < 0:
            e[n:, :m] = xs[:-n, -m:]
        elif n < 0 and m > 0:
            e[:n, m:] = xs[-n:, :-m]
        elif n < 0 and m < 0:
            e[:n, :m] = xs[-n:, -m:]
        else:
            e = xs
        return e


    #image = d.get_batch()

    #print(image.shape)
    #temp = image[:,0,:,:]
    #j = 4
    #image[:,1,:,:] = np.zeros((5, 256,256))
    #image[:,1,:128-j,:] = temp[:,j:128, :]
    #image[:,

    def get_eval(self,):
        c = 200
        width = 128
        i = 4
        image = self.test_data[c:c+width,c:c+width,i]/256.0
        target = self.test_data[c:c+width,c:c+width,i+1]/256.0

        image = cl.image_processing.resize(image, (2, 2), order=2)
        target = cl.image_processing.resize(target, (2, 2), order=2)
        image = np.tile(image,(self.batch_size,1,1))
        target = np.tile(target,(self.batch_size,1,1))
        label = np.zeros((self.batch_size, 256, 256), dtype=np.float32)
        print(image.shape, target.shape, label.shape)
        return image, target, label

if __name__ == "__main__":
    image = np.ones((5,8,256,256))
    image = Data.augmentation(image)
    cl.visual.save(image[4,5], 'dump/image')
