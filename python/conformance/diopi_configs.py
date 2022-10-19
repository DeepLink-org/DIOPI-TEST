from .config import Genfunc
from .dtype import Dtype

diopi_configs = {
    'batch_norm': dict(
        name=["batch_norm"],
        dtype=[Dtype.float32],
        atol=1e-5,
        para=dict(
            training=[False, False, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": ((8, ), (64, ), None, (16, )),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    "shape": ((8, ), (64, ), None, (16, )),
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    "shape": ((8, ), (64, ), (96, ), (16, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        atol=1e-4,
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

    'relu': dict(
        name=["relu"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 4096), (64, 28, 28),
                              (32, 64, 112, 112), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
