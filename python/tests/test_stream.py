import numpy as np
from threading import Thread
from conformance.diopi_runtime import Context, Tensor, Sizes
from conformance.diopi_runtime import Device
from conformance.dtype import Dtype


class TestStream(object):
    context = Context()
    context1 = Context()
    stream = context.get_handle()
    stream1 = context1.get_handle()

    def check_get_device_data(self, stream):
        res_tensor = Tensor([], Dtype.float32, context_handle=stream)
        assert res_tensor.get_device() == Device.AIChip

    def test_stream(self):
        self.check_get_device_data(self.stream)

    def test_multi_stream(self):
        self.check_get_device_data(self.stream)
        self.check_get_device_data(self.stream1)

    def test_multi_thread_multi_stream(self):
        thread_1 = Thread(target=self.check_get_device_data, args=(self.stream, ))
        thread_2 = Thread(target=self.check_get_device_data, args=(self.stream1, ))
        thread_1.start()
        thread_2.start()
        thread_1.join()
        thread_2.join()
