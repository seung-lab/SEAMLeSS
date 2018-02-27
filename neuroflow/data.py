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
                                batch_size = self.batch_size,
                                features=hparams.features,
                                flipping=hparams.augmentation["flipping"],
                                random_brightness=hparams.augmentation["random_brightness"],
                                random_elastic_transform=hparams.augmentation["random_elastic_transform"])
        self.similar = True
        self.test_data = np.load('data/evaluate/simple_test.npy').astype(np.float32)

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

    def get_batch(self):

        image, label, target  = self.data.get_batch()
        if not self.check_validity(image, target):
            return self.get_batch()

        label += image == 0 # Borders

        label = (label>0).astype(np.float32)

        return image, target, label

    def dissimilar(self, images, templates):
        length = templates.shape[0]-1
        temp = np.array(templates[0])
        templates[0:length] = templates[1:length+1]
        templates[length] = temp
        return images, templates

    def check_validity(self, image, template):
        t = np.array(template.shape)
        if np.any(np.sum(image<0.05, axis=(1,2)) >= t[1]*t[2]) or image.shape[0]<self.batch_size:
            return False

        #if np.any(template.var(axis=(1,2))<0.0001):
        #    print("hey")
        #    return False

        return True
