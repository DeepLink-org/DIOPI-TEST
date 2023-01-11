DIOPI 测例说明
===================

配置文件规则
------------------------
.. code-block:: python

    'conv_2d': dict(
        name=["conv2d"],
        atol=1e-3,
        rtol=1e-3,
        dtype=[Dtype.float32, Dtype.float16],
        para=dict(
            stride=[2, 1, 1],
            padding=[0, 12, 0],
            dilation=[1, 12, 1],
            groups=[1, 2048, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((2, 256, 200, 304), (2, 2048, 64, 64), (2, 2048, 1, 1)),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((12, 256, 1, 1), (2048, 1, 3, 3), (512, 2048, 1, 1)),
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": ((12, ), None, None),
                },
            ]
        ),
    ),

* name: *list*
    函数名字。如conv2d 在生成基准数据中使用到的函数名字：

    torch.nn.functional.conv2d(*input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1*) -> Tensor

    在测试算子适配中使用到的 python 函数名字：

    diopi_funtions.conv2d(*input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1*) -> Tensor

* atol: *float*
    用于结果验证的绝对误差参数。
* rtol: *float*
    用于结果验证的相对误差参数。
    结果验证使用此函数检查 input 和 other 是否满足条件：

    \|input - other\| ≤ atol + rtol x \|other\|
* dtype: *list*
    函数张量参数的数据类型。
* tensor_para: *dict*
    函数张量参数（比如 input, weight, bias 等）。
    args 中：
    ins: *list* 张量参数的名字。

    shape: *tuple* 张量参数的形状。

    gen_fn (缺省): *builtin_function*
    数据生成器，使用了 numpy.random.randn 作为默认生成器。
    
    args 中包含`多组`测试用例张量, shape 元素个数代表张量组数，每个 ins 的 shape 都是`一一对应`的。
    conv_2d 中描述了三组输入张量测试用例：

    第一组：`group0 = {"input": tensor(2, 256, 200, 304), "weight": tensor(12, 256, 1, 1), "bias": tensor(12),}`
    
    第二组：`group1 = {"input": tensor(2, 2048, 64, 64), "weight": tensor(2048, 1, 3, 3), "bias": None,}`
    
    第三组：`group2 = {"input": tensor(2, 2048, 1, 1), "weight": tensor(512, 2048, 1, 1), "bias": None,}`
* para: *dict*


可选测试模式
------------------------
* fname: 指定算子测试
* filter_dtype: 过滤指定数据类型的测试
* nhwc: 使用NHWC格式的张量测试
* four_bytes: 使用int32代替int64测试
* model_name: 指定模型相关算子测试

反向测试规则
------------------------
* 反向测试算子范围
* 反向测试基准数据
* 反向测试运行机制

