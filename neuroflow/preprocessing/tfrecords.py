import json
import cavelab as cl
import tensorflow as tf
import numpy as np
import time

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_points(path):
    with open(hparams.points, 'r') as f:
        points = json.load(f)
    return points

def get_random_points(bbox, n): #[[x1,x2], [y1,y2], [z1,z2]]
    points = np.random.rand(n, 3)
    for i in range(3):
        points[:,i] = points[:,i]*(bbox[i][1]-bbox[i][0])+bbox[i][0]
    return points.astype(np.int).tolist()

def get_cloudvolumes(src, start_mip, end_mip):
    volume_mip = []
    for i in range(start_mip, end_mip):
        volume_mip.append(cl.Cloud(src, mip=i, cache=False))
    return volume_mip

def normalize(image, var = 3, axis=(0,1,2)): # [d,width, height,b]
    mask = image==0
    image = np.ma.array(image, mask=mask, dtype=np.float32)
    mean = image.mean(axis, keepdims=True)
    std = image.std(axis, keepdims=True)
    image = (image-mean)/std

    black = -var*(image<-var)
    white = var*(image>var)
    image = np.multiply(image, np.abs(image)<var)+black+white
    mn = -var
    mx = var
    image = (image-mn)/(mx-mn)
    return image.data.astype(np.float32)

def create_tf_records(hparams, train=True):
    lvls, dpth, size = hparams.levels, hparams.depth,  hparams.width
    path, sign = hparams.tfrecord_train_dest, -1
    if not train:
        path, sign = hparams.tfrecord_test_dest, 1

    #points = get_points(hparams.points)
    points = get_random_points([[50000, 240000], [70000, 308815], [1, 230]], 30000)

    writer = tf.python_io.TFRecordWriter(path)
    vols = get_cloudvolumes(hparams.cloud_src, hparams.cloud_mip, hparams.cloud_mip+lvls)
    count = 0
    for point in points:
        t1 = time.time()
        if(sign*(point[-1]-140)>0):
            continue
        image = np.zeros((lvls, size, size, dpth))
        try:
            for i in range(lvls):
              _size = int(size*(2**(i+2)))
              (x,y) = (int(point[0] - int(_size/2)), int(point[1] - int(_size/2)))
              image[i] = vols[i].read_global((x,y,point[-1]-5),(size,size, dpth))
        except:
            continue
        image = image/255.0
        image = normalize(image)
        image = 255.0*image
        image_raw = np.asarray(image, dtype=np.uint8).tostring()
        ex = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image_raw),
        }))
        writer.write(ex.SerializeToString())
        count += 1
        t2 = time.time()
        print(count, t2-t1)

    writer.close()

if __name__ == "__main__":
    hparams = cl.hparams(name="preprocessing")
    create_tf_records(hparams, train=True)
    #create_tf_records(hparams, train=False)
