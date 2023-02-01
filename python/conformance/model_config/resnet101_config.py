from ..config import Genfunc
from ..dtype import Dtype

# source resnet101_8xb32_in1k.py
resnet50_config = {
    'conv_2d': dict(
        name=["conv2d"],
        atol=1e-3,
        rtol=1e-3,
        dtype=[Dtype.float32],
        para=dict(
            stride=[(2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)],
            padding=[(3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((32, 3, 224, 224), (32, 64, 56, 56), (32, 64, 56, 56), (32, 64, 56, 56), (32, 256, 56, 56), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28), (32, 256, 56, 56), (32, 512, 28, 28), (32, 512, 28, 28), (32, 256, 28, 28), (32, 256, 14, 14), (32, 512, 28, 28), (32, 1024, 14, 14), (32, 1024, 14, 14), (32, 512, 14, 14), (32, 512, 7, 7), (32, 1024, 14, 14), (32, 2048, 7, 7)),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1)),
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
                },
            ]
        ),
    ),

    'batch_norm': dict(
        name=["batch_norm"],
        dtype=[Dtype.float32],
        atol=1e-3,
        rtol=1e-3,
        para=dict(
            training=[True, True, True, True, True, True, True, True, True, True, True, True],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((32, 64, 112, 112), (32, 64, 56, 56), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28), (32, 512, 28, 28), (32, 256, 28, 28), (32, 256, 14, 14), (32, 1024, 14, 14), (32, 512, 14, 14), (32, 512, 7, 7), (32, 2048, 7, 7)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": ((64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    "shape": ((64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)),
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    "shape": ((64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((32, 64, 112, 112), (32, 64, 56, 56), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28), (32, 512, 28, 28), (32, 256, 28, 28), (32, 256, 14, 14), (32, 1024, 14, 14), (32, 512, 14, 14), (32, 512, 7, 7), (32, 2048, 7, 7)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[3],
            stride=[2],
            padding=[1],
            dilation=[1],
            ceil_mode=[False],
            return_indices=[False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((32, 64, 112, 112), ),
                    "dtype": [Dtype.float32],
                },
            ]
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[(1, 1)],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": [(32, 2048, 7, 7)],
                    "dtype": [Dtype.float32],
                },
            ]
        ),
    ),

    'linear': dict(
        name=["linear"],
        atol=1e-4,
        rtol=1e-5,
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((32, 2048), ),
                },
                {
                    "ins": ['weight'],
                    "requires_grad": [True],
                    "shape": ((1000, 2048), ),
                },
                {
                    "ins": ['bias'],
                    "requires_grad": [True],
                    "shape": ((1000,), ),
                },
            ]
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        atol=1e-1,
        rtol=1e-2,
        para=dict(
            reduction=['none'],
        ),
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": [(32, 1000)],
                },
                {
                    "ins": ['target'],
                    "shape": [(32,)],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=1000),
                    "dtype": [Dtype.int64],

                },
            ],
        ),
    ),

    'add_1': dict(
        name=['add'],
        interface=['torch'],
        dtype=[Dtype.float32],
        para=dict(
            alpha=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, 3, 7, 7), (64,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (256,), (64, 256, 1, 1), (128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (1000, 2048), (1000,)),
                },
                {
                    "ins": ['other'],
                    "shape": ((64, 3, 7, 7), (64,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (256,), (64, 256, 1, 1), (128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (1000, 2048), (1000,)),
                },
            ],
        ),
    ),

    'add_2': dict(
        name=['add'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32],
        para=dict(
            alpha=[-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, 3, 7, 7), (64,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (256,), (64, 256, 1, 1), (128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (1000, 2048), (1000,)),
                },
                {
                    "ins": ['other'],
                    "shape": ((64, 3, 7, 7), (64,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (256,), (64, 256, 1, 1), (128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (1000, 2048), (1000,)),
                },
            ],
        ),
    ),

    'add_3': dict(
        name=['add'],
        interface=['torch'],
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((32, 256, 56, 56), (32, 512, 28, 28), (32, 1024, 14, 14), (32, 2048, 7, 7)),
                },
                {
                    "ins": ['other'],
                    "shape": ((32, 256, 56, 56), (32, 512, 28, 28), (32, 1024, 14, 14), (32, 2048, 7, 7)),
                },
            ],
        ),
    ),

    'add_4': dict(
        name=['add'],
        interface=['torch'],
        dtype=[Dtype.float32],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,)),
                },
            ],
        ),
    ),

    'mul': dict(
        name=['mul'],
        interface=['torch'],
        dtype=[Dtype.float32],
        para=dict(
            other=[1.0],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,)),
                },
            ],
        ),
    ),

    'div': dict(
        name=['div'],
        interface=['torch'],
        dtype=[Dtype.float32],
        para=dict(
            other=[32],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,)),
                },
            ],
        ),
    ),

    'sum': dict(
        name=['sum'],
        interface=['torch'],
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((32,),),
                },
            ],
        ),
    ),

    'mean': dict(
        name=['mean'],
        interface=['torch'],
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,),),
                },
            ],
        ),
    ),

   'randperm': dict(
        name=['randperm'],
        no_output_ref=True,
        para=dict(
            n=[1281167],
        ),
    ),

    'sgd': dict(
        name=["sgd"],
        interface=["CustomizedTest"],
        para=dict(
            nesterov=[False for i in range(28)],
            lr=[0.1 for i in range(28)],
            momentum=[0.9 for i in range(28)],
            weight_decay=[0.0001 for i in range(28)],
            dampening=[0 for i in range(28)],
        ),
        tensor_para=dict(
            dtype=[Dtype.float32],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [(64, 3, 7, 7), (64,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (256,), (64, 256, 1, 1), (128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (1000, 2048), (1000,)],
                    "gen_fn": Genfunc.rand,
                },
                {
                    "ins": ['buf'],
                    "shape": [(64, 3, 7, 7), (64,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (256,), (64, 256, 1, 1), (128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (1000, 2048), (1000,)],
                    "gen_fn": Genfunc.zeros,
                },
            ]
        ),
    ),
}