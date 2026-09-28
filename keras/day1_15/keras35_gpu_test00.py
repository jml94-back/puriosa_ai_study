import tensorflow as tf
print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus) #[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

if(gpus):
    print("GPU 있다!~!")
else:
    print("GPU 없다!!!")

virtual_devices = tf.config.get_logical_device_configuration(gpus[0])
        
if virtual_devices:
    print("현재 설정된 Virtual Device 구성:")
    for idx, dev in enumerate(virtual_devices):
        print(f"  - Virtual GPU {idx}: Memory Limit = {dev.memory_limit} MB")
else:
    print("별도의 Virtual Device Limit(메모리 제한)이 설정되어 있지 않습니다.")