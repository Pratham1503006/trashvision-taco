import time
import tensorflow as tf
print("TF", tf.__version__, flush=True)
cfg = tf.ConfigProto(log_device_placement=False)
with tf.Session(config=cfg) as sess:
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])
    c = tf.matmul(a, b)
    t0 = time.time()
    sess.run(c)
    print("first matmul:", round(time.time() - t0, 2), "s", flush=True)
    t0 = time.time()
    for _ in range(10):
        sess.run(c)
    print("10x matmul:", round(time.time() - t0, 2), "s", flush=True)
