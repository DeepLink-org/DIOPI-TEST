from .config import Genfunc
from .dtype import Dtype

diopi_configs = {
    'batch_norm': dict(
        name=["batch_norm"],
        dtype=[Dtype.float32],
        atol=1e-2,
        para=dict(
            training=[True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((32, 64, 112, 112), (32, 64, 56, 56), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28),
                    (32, 512, 28, 28), (32, 256, 28, 28), (32, 256, 14, 14), (32, 1024, 14, 14), (32, 512, 14, 14),
                    (32, 512, 7, 7), (32, 2048, 7, 7)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": ((64, ), (64, ), (256, ), (128, ), (128, ),
                    (512, ), (256, ), (256, ), (1024, ), (512, ),
                    (512, ), (2048, )),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    "shape": ((64, ), (64, ), (256, ), (128, ), (128, ),
                    (512, ), (256, ), (256, ), (1024, ), (512, ),
                    (512, ), (2048, )),
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    "shape": ((64, ), (64, ), (256, ), (128, ), (128, ),
                    (512, ), (256, ), (256, ), (1024, ), (512, ),
                    (512, ), (2048, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        atol=1e-3,
        rtol=1e-3,
        dtype=[Dtype.float32, Dtype.float16],
        para=dict(
            stride=  [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2],
            padding= [3, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ,0, 0, 1, 0, 0, 1,
                      0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
                      0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 3],
            dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],#[1, 12, 1],
            groups=  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],#[1, 2048, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((32, 3, 224, 224), (32, 64, 56, 56), (32, 64, 56, 56), (32, 64, 56, 56), (32, 64, 56, 56),
                    (32, 256, 56, 56), (32, 64, 56, 56), (32, 64, 56, 56), (32, 256, 56, 56), (32, 64, 56, 56),
                    (32, 64, 56, 56), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28), (32, 256, 56, 56), 
                    (32, 512, 28, 28), (32, 128, 28, 28), (32, 128, 28, 28), (32, 512, 28, 28), (32, 128, 28, 28),
                    (32, 128, 28, 28), (32, 512, 28, 28), (32, 128, 28, 28), (32, 128, 28, 28), (32, 512, 28, 28),
                    (32, 256, 28, 28), (32, 256, 14, 14), (32, 512, 28, 28), (32, 1024, 14, 14), (32, 256, 14, 14),
                    (32, 256, 14, 14), (32, 1024, 14, 14), (32, 256, 14, 14), (32, 256, 14, 14), (32, 1024, 14, 14),
                    (32, 256, 14, 14), (32, 256, 14, 14), (32, 1024, 14, 14), (32, 256, 14, 14), (32, 256, 14, 14), 
                    (32, 1024, 14, 14), (32, 256, 14, 14), (32, 256, 14, 14), (32, 1024, 14, 14), (32, 512, 14, 14),
                    (32, 512, 7, 7), (32, 1024, 14, 14), (32, 2048, 7, 7), (32, 512, 7, 7), (32, 512, 7, 7), 
                    (32, 2048, 7, 7), (32, 512, 7, 7), (32, 512, 7, 7), (32, 3, 224, 224), ),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3),(256, 64, 1, 1), (256, 64, 1, 1),
                    (64, 256, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (64, 64, 3, 3),
                    (256, 64, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1),
                    (128, 512, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3),
                    (512, 128, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (256, 512, 1, 1),
                    (256, 256 ,3 ,3), (1024, 256, 1 ,1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), 
                    (1024, 256, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (256, 1024, 1, 1),
                    (256, 256, 3, 3), (1024, 256, 1, 1), (256, 1024, 1, 1), (256, 256, 3 ,3), (1024, 256, 1, 1),
                    (256, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3),
                    (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3 ,3), (2048, 512, 1, 1),
                    (512, 2048, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (64, 3, 7, 7)),
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": ((64,), (64,), (64,), (256,), (256,),
                    (64,), (64,), (256,), (64,), (64,),
                    (256,), (128,), (128,), (512,), (512,),
                    (128,), (128,), (512,), (128,), (128,),
                    (512,), (128,), (128,), (512,), (256,),
                    (256,), (1024,), (1024,), (256,), (256,),
                    (1024,), (256,), (256,), (1024,), (256,),
                    (256,), (1024,), (256,), (256,), (1024,),
                    (256,), (256,), (1024,), (512,), (512,),
                    (2048,), (2048,), (512,), (512,), (2048,),
                    (512,), (512,), (2048,), (64,),),
                },
            ]
        ),
    ),

    'relu': dict(
        name=["relu"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((32, 64, 112, 112), (32, 64, 56, 56), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28),
                    (32, 512, 28, 28), (32, 256, 28, 28), (32, 256, 14, 14), (32, 1024, 14, 14), (32, 512, 14, 14),
                    (32, 512, 7, 7), (32, 2048, 7, 7)),
                    "dtype": [Dtype.float32, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        atol=1e-5,
        rtol=1e-4,
        atol_half=1e-3,
        rtol_half=1e-2,
        para=dict(
            output_size=[(1, 1)],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((32, 2048, 7, 7), ),
                    "dtype": [Dtype.float32, Dtype.float16],
                },
            ]
        ),
    ),

    'addmm': dict(
        name=["addmm"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            alpha=[0.001, -0.01, 1],
            beta=[0.001, -0.01, 1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float32, Dtype.float16],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((10, ), (768, ), (400,)),
                },
                {
                    "ins": ["mat1"],
                    "shape": ((2, 2048), (2, 768), (1, 2304)),
                },
                {
                    "ins": ["mat2"],
                    "shape": ((2048, 10), (768, 768), (2304, 400)),
                },
            ],
        ),
    ),

    'nll_loss': dict(
        name=["nll_loss"],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            reduction=['none', 'mean', 'sum'],
            ignore_index=[-100, 92, 255],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((200, 81), (2, 92, 29), (2, 150, 512, 512)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['target'],
                    "shape": ((200, ), (2, 29), (2, 512, 512)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=80),
                },
                {
                    "ins": ['weight'],
                    "shape": ((81, ), (92, ), None),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.ones,
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            reduction=['mean', 'none'],
            ignore_index=[0, -100],
            label_smoothing=[0.0, 0.5],
        ),
        dtype=[Dtype.float32, Dtype.float16],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, 81), (3, 5, 6, 6)),
                },
                {
                    "ins": ['weight'],
                    "shape": (None, (5,)),
                },
                {
                    "ins": ['target'],
                    "shape": ((1024, ), (3, 6, 6)),
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=5),
                    "dtype": [Dtype.int64],

                },
            ],
        ),
    ),

    'linear': dict(
        name=["linear"],
        atol=1e-4,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 512), (128, 49, 128), (6, 2, 100, 256),
                              (2, 31, 6, 40, 512)),
                    "dtype": [Dtype.float32, Dtype.float16],
                },
                {
                    "ins": ['weight'],
                    "shape": ((10, 512), (384, 128), (81, 256), (1, 512)),
                    "dtype": [Dtype.float32, Dtype.float16],
                },
                {
                    "ins": ['bias'],
                    "shape": ((10, ), None, (81, ), (1,)),
                    "dtype": [Dtype.float32, Dtype.float16],
                },
            ]
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((78, 24), (2, 92, 29), (2, 150, 512, 512)),
                    "dtype": [Dtype.float32, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sgd': dict(
        name=["sgd"],
        interface=["CustomizedTest"],
        atol_half=1e-4,
        rtol_half=1e-3,
        para=dict(
            nesterov=[False, True],
            lr=[0.1, 0.1],
            momentum=[0.01, 0.01],
            weight_decay=[0, 0.1],
            dampening=[0.1, 0],
        ),
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float16],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [(2, 3, 16), (4, 32, 7, 7)],
                    "gen_fn": Genfunc.rand,
                },
                {
                    "ins": ['buf'],
                    "shape": [(2, 3, 16), (4, 32, 7, 7)],
                    "gen_fn": Genfunc.zeros,
                },
            ]
        ),
    ),

   'randperm': dict(
        name=['randperm'],
        no_output_ref=True,
        para=dict(
            n=[2, 1999, 640000],
        ),
    ),

   'random': dict(
        name=['random'],
        no_output_ref=True,
        para={
            'start':[0, 3, -1, 0],
            'end':[2, None, 1, None],
        },
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (64, 64), (16, 1, 3, 3), (96, 48, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.int64, Dtype.float16],
                },
            ],
        ),
    ),

    'pointwise_binary_scalar': dict(
        name=['add', 'mul', 'div', 'eq',
              'ne', 'le',  'lt', 'gt', 'ge'],
        interface=['torch'],
        dtype=[Dtype.float32, Dtype.float16],
        para=dict(
            other=[-1, 0.028, 2, 1.0],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'reduce_op': dict(
        name=['mean', 'sum'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float16],
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
                    "shape": ((32, 64, 112, 112),),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),
}
