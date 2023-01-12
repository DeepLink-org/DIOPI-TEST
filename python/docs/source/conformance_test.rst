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

1. 反向测试算子范围
~~~~~~~~~~~~~~~~~~~~~~~~

      并非所有算子都要进行反向测试, 这是因为部分 DIOPI 算子并没有对应的反向算子声明。譬如 DiopiAdd, 
      因为 add 算子的反向也通过 add 实现, 故没有必要声明一个类似于 DiopiAddBackward 算子。
      常见训练框架实现自动微分时, 也一般是复用 add 算子计算反向。另外, 即使训练框架真的定义了一个叫 add_backward
      的函数, 在框架适配 DIOPI 算子时，我们也只需要将 DiopiAdd 包装进 add_backward 即可。

      具体哪些算子需要测试反向, 可以通过 diopirt/include/diopi/functions.h 中函数声明查询, 若存在 DIOPI 反向算子声明,
      则该算子一定会有相应的反向测试。
      另外, 也可以查询 python/conformance/diopi_configs.py 文件, 若对 tensor_para 中 args 之一的张量将其 requires_grad
      属性设置为 True, 则该算子会同时测试其相应的反向算子。以 log_softmax 的测例配置为例： 

      .. code-block:: python

        'log_softmax': dict(
            name=["log_softmax"],
            saved_args=dict(output=0), # 指定反向算子需要的第 x 个前向输出结果
            para=dict(
                dim=[-1, 1, 0],
            ),
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "requires_grad": [True], # requires_grad 为 True 则需要反向测试
                        "shape": ((78, 24), (2, 92, 29), (2, 150, 512, 512)),
                        "dtype": [Dtype.float32, Dtype.float64],
                        "gen_fn": Genfunc.randn,
                    },
                ],
            ),
        ),

2. 反向测试基准数据
~~~~~~~~~~~~~~~~~~~~~~~~

    * **反向测试的基准输出数据** 使用 pytorch 的 torch.autograd 接口自动在每个前向算子完成时进行反向计算,
      并将计算结果保存下来作为基准输出数据。另外, 初始回传梯度通过 ones_like 生成, 初始梯度与输出大小相同但值均为 1。
      如果前向算子有多个输出, 可以通过 diopi_configs.py 配置文件中的 requires_backward 的值, 如指定
      requires_backward=[0], 则只对第 1 个输出结果张量创建梯度并回传。目前暂无算子测例使用 requires_backward 属性。

    .. code-block:: python

        class GenOutputData(object):
            r'''
            Generate output data for all functions by using torch and input data
            '''
            @staticmethod
            def run(func_name, model_name, filter_dtype_str_list):
                ...
                for saved_pth in saved_pth_list: # 循环每个算子测例
                    ...
                    if function_paras["requires_grad"]: # 判断是否需要反向测试
                        ...
                        # 若未指定 requires_backward 则对所有前向输出结果张量创建梯度
                        # 否则, 仅对指定前向输出结果张量创建梯度
                        requires_backward = data["cfg"]["requires_backward"]
                        outputs_for_backward = outputs if len(requires_backward) == 0 \
                        else [outputs[i] for i in requires_backward]

                        inputs_name_for_grad, inputs_for_grad = get_name_and_data_for_grad(function_paras)
                        saved_grads = None
                        if len(inputs_for_grad) != 0:
                            # 通过 ones_like 函数创建初始梯度
                            grad_outputs = [torch.ones_like(i) for i in outputs_for_backward]
                            # 通过 torch.autograd.grad 自动微分进行反向计算, 得到反向基准输出数据
                            grads = torch.autograd.grad(
                                outputs_for_backward, inputs_for_grad, grad_outputs, allow_unused=True)
                            saved_grads = {k: v for k, v in zip(inputs_name_for_grad, grads)}

    * **反向测试的基准输入数据** 主要是复用前向的输入参数和以及指定的输出结果。以上述 log_softmax 为例, 在调用 python 层反向算子时, 会将所有的前向参数
      dim, input 传入 python 层反向算子。 另外如果指定了 saved_args, 还需要传递 saved_args 指定的前向输出结果。如 log_softmax 测例指定了
      saved_args=dict(output=0), 且 log_softmax 只返回一个输出, 故这里会将第一个输出也是唯一的输出传递给反向算子。

      另外有些前向参数可能不被反向计算所需要, 这里是通过 \*\*kwargs 不指定关键字参数个数来处理。这是因为一致性测试框架主要以键值对的方式传参到 
      python/conformance/diopi_functions.py 中的 python 函数接口进行测试。我们在定义反向函数接口时,
      会添加一个 \*\*kwargs 参数来接受不被使用的关键字参数。

      .. code-block:: python

        def log_softmax(input, dim, dtype=None):
            ...

        # 所有 python 层反向算子接口均以前向函数名加上 _backward 命名
        # 所有 python 层反向算子接口均有 **kwargs 参数以接受不定长且不被使用的前向算子参数
        def log_softmax_backward(input, grad_outputs, output, dim, **kwargs):
            ...

3. 反向测试运行机制
~~~~~~~~~~~~~~~~~~~~~~~~

    - 在 diopi_configs.py 配置文件中为有反向声明的 DIOPI 算子通过指定输入张量的 requires_grad
      属性为 True 来表示需要进行反向测试

    - 反向测试打包所有前向参数以及 saved_args 中指定的某个前向输出结果到 python 反向函数接口。
      在 diopi_functions.py 封装的函数中, 反向函数以前向函数名加上 _backward 命名,
      另外添加 \*\*kwargs 来接受不定长的关键字参数。传参逻辑如下：

    .. code-block:: python

        class ConformanceTest(object):
            r'''
            Run all functions by using input, then compare_with_gen_output with saved output
            '''
            @staticmethod
            def run(func_name, model_name, filter_dtype_str_list):
                ...
                for saved_pth in saved_pth_list: # 循环每个算子测例
                    ...
                    # 判断是否需要反向测试
                    if function_paras["requires_grad"] and "inplace=True" not in func_call:
                        ...
                        # requires_backward 作用同上，用以创建指定输出张量的梯度
                        requires_backward = data["cfg"]["requires_backward"]
                        outputs_for_backward = output if len(requires_backward) == 0 \
                            else [output[i] for i in requires_backward]

                        backward_para = {}
                        grad_outputs = [F.ones_like(i) for i in outputs_for_backward]
                        backward_para["grad_outputs"] = grad_outputs
                        # 将 saved_args 中指定的前向输出存在 backward_para 字典中
                        for k, v in data["cfg"]['saved_args'].items():
                            backward_para[k] = output[v]

                        try:
                            # 将所有前向算子的关键字参数以及 backward_para 打包传递给反向算子
                            grad_input = eval(f"F.{cfg_func_name}_backward(**kwargs, **backward_para)")
                        ...
        
    - 在一致性测试框架中计算反向结果, 并同基准输出数据对比。