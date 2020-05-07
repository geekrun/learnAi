import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  # 输出可用GPU数量
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))  # 输出可用CPU数量
tf.test.is_gpu_available()  # 输出当前是否正在使用GPU，不出意外应该是True
