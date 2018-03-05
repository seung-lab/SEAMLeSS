import json
import cavelab as cl
import tensorflow as tf
import numpy as np

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_points(path):
    with open(hparams.points, 'r') as f:
        points = json.load(f)
    return points

def get_cloudvolumes(src, start_mip, end_mip):
    volume_mip = []
    for i in range(start_mip, end_mip):
        volume_mip.append(cl.Cloud(src, mip=i, cache=False))
    return volume_mip

def create_tf_records(hparams, train=True):
    lvls, dpth, size = hparams.levels, hparams.depth,  hparams.width
    path, sign = hparams.tfrecord_train_dest, -1
    if not train:
        path, sign = hparams.tfrecord_test_dest, 1

    points = get_points(hparams.points)
    writer = tf.python_io.TFRecordWriter(path)
    vols = get_cloudvolumes(hparams.cloud_src, hparams.cloud_mip, hparams.cloud_mip+lvls)
    count = 0
    for point in points:
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
        image_raw = np.asarray(image, dtype=np.uint8).tostring()
        ex = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image_raw),
        }))
        writer.write(ex.SerializeToString())
        count += 1
        print(count)

    writer.close()

if __name__ == "__main__":
    hparams = cl.hparams(name="preprocessing")
    create_tf_records(hparams, train=True)
    create_tf_records(hparams, train=False)
