import numpy as np
import tensorflow as tf

def gen():
    while True:
        yield np.random.randint(0, 10)

ds = tf.data.Dataset.from_generator(
    gen,
    output_types=(tf.int32),
    output_shapes=(()))

ds

ds_batch = ds.shuffle(10).batch(20)
print(ds_batch)
