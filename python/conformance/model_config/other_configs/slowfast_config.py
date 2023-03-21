from ...config import Genfunc
from ...dtype import Dtype

slowfast_config = {
    'randperm': dict(
        name=["randperm"],
        no_output_ref=True,
        para=dict(
            n=[86017],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        para=dict(
            scale_factor=[(0.25, 1.0, 1.0), (1.0, 1.0, 1.0)],
            mode=['nearest', 'nearest'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 3, 64, 224, 224), (4, 3, 64, 224, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'conv3d': dict(
        name=["conv3d"],
        para=dict(
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(1, 2, 2), (1, 2, 2), (4, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (4, 1, 1), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 1, 1), (4, 1, 1), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 1, 1), (4, 1, 1), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 1, 1), (1, 1, 1)],
            padding=[(0, 3, 3), (2, 3, 3), (3, 0, 0), (0, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 1), (0, 0, 0), (1, 0, 0), (3, 0, 0), (0, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 1), (1, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 1), (3, 0, 0), (1, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 1), (3, 0, 0), (1, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 1)],
            dilation=[(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 3, 16, 224, 224), (4, 3, 64, 224, 224), (4, 8, 64, 56, 56), (4, 80, 16, 56, 56), (4, 64, 16, 56, 56), (4, 64, 16, 56, 56), (4, 80, 16, 56, 56), (4, 256, 16, 56, 56), (4, 8, 64, 56, 56), (4, 8, 64, 56, 56), (4, 8, 64, 56, 56), (4, 32, 64, 56, 56), (4, 32, 64, 56, 56), (4, 320, 16, 56, 56), (4, 128, 16, 56, 56), (4, 128, 16, 28, 28), (4, 320, 16, 56, 56), (4, 512, 16, 28, 28), (4, 128, 16, 28, 28), (4, 32, 64, 56, 56), (4, 16, 64, 56, 56), (4, 16, 64, 28, 28), (4, 32, 64, 56, 56), (4, 64, 64, 28, 28), (4, 16, 64, 28, 28), (4, 64, 64, 28, 28), (4, 640, 16, 28, 28), (4, 256, 16, 28, 28), (4, 256, 16, 14, 14), (4, 640, 16, 28, 28), (4, 1024, 16, 14, 14), (4, 256, 16, 14, 14), (4, 64, 64, 28, 28), (4, 32, 64, 28, 28), (4, 32, 64, 14, 14), (4, 64, 64, 28, 28), (4, 128, 64, 14, 14), (4, 32, 64, 14, 14), (4, 128, 64, 14, 14), (4, 1280, 16, 14, 14), (4, 512, 16, 14, 14), (4, 512, 16, 7, 7), (4, 1280, 16, 14, 14), (4, 2048, 16, 7, 7), (4, 512, 16, 7, 7), (4, 128, 64, 14, 14), (4, 64, 64, 14, 14), (4, 64, 64, 7, 7), (4, 128, 64, 14, 14), (4, 256, 64, 7, 7), (4, 64, 64, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 1, 7, 7), (8, 3, 5, 7, 7), (16, 8, 7, 1, 1), (64, 80, 1, 1, 1), (64, 64, 1, 3, 3), (256, 64, 1, 1, 1), (256, 80, 1, 1, 1), (64, 256, 1, 1, 1), (8, 8, 3, 1, 1), (8, 8, 1, 3, 3), (32, 8, 1, 1, 1), (8, 32, 3, 1, 1), (64, 32, 7, 1, 1), (128, 320, 1, 1, 1), (128, 128, 1, 3, 3), (512, 128, 1, 1, 1), (512, 320, 1, 1, 1), (128, 512, 1, 1, 1), (128, 128, 1, 3, 3), (16, 32, 3, 1, 1), (16, 16, 1, 3, 3), (64, 16, 1, 1, 1), (64, 32, 1, 1, 1), (16, 64, 3, 1, 1), (16, 16, 1, 3, 3), (128, 64, 7, 1, 1), (256, 640, 3, 1, 1), (256, 256, 1, 3, 3), (1024, 256, 1, 1, 1), (1024, 640, 1, 1, 1), (256, 1024, 3, 1, 1), (256, 256, 1, 3, 3), (32, 64, 3, 1, 1), (32, 32, 1, 3, 3), (128, 32, 1, 1, 1), (128, 64, 1, 1, 1), (32, 128, 3, 1, 1), (32, 32, 1, 3, 3), (256, 128, 7, 1, 1), (512, 1280, 3, 1, 1), (512, 512, 1, 3, 3), (2048, 512, 1, 1, 1), (2048, 1280, 1, 1, 1), (512, 2048, 3, 1, 1), (512, 512, 1, 3, 3), (64, 128, 3, 1, 1), (64, 64, 1, 3, 3), (256, 64, 1, 1, 1), (256, 128, 1, 1, 1), (64, 256, 3, 1, 1), (64, 64, 1, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'batch_norm': dict(
        name=["batch_norm"],
        para=dict(
            training=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 64, 16, 112, 112), (4, 8, 64, 112, 112), (4, 64, 16, 56, 56), (4, 256, 16, 56, 56), (4, 8, 64, 56, 56), (4, 32, 64, 56, 56), (4, 128, 16, 56, 56), (4, 128, 16, 28, 28), (4, 512, 16, 28, 28), (4, 16, 64, 56, 56), (4, 16, 64, 28, 28), (4, 64, 64, 28, 28), (4, 256, 16, 28, 28), (4, 256, 16, 14, 14), (4, 1024, 16, 14, 14), (4, 32, 64, 28, 28), (4, 32, 64, 14, 14), (4, 128, 64, 14, 14), (4, 512, 16, 14, 14), (4, 512, 16, 7, 7), (4, 2048, 16, 7, 7), (4, 64, 64, 14, 14), (4, 64, 64, 7, 7), (4, 256, 64, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": [(64,), (8,), (64,), (256,), (8,), (32,), (128,), (128,), (512,), (16,), (16,), (64,), (256,), (256,), (1024,), (32,), (32,), (128,), (512,), (512,), (2048,), (64,), (64,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": [(64,), (8,), (64,), (256,), (8,), (32,), (128,), (128,), (512,), (16,), (16,), (64,), (256,), (256,), (1024,), (32,), (32,), (128,), (512,), (512,), (2048,), (64,), (64,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64,), (8,), (64,), (256,), (8,), (32,), (128,), (128,), (512,), (16,), (16,), (64,), (256,), (256,), (1024,), (32,), (32,), (128,), (512,), (512,), (2048,), (64,), (64,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (8,), (64,), (256,), (8,), (32,), (128,), (128,), (512,), (16,), (16,), (64,), (256,), (256,), (1024,), (32,), (32,), (128,), (512,), (512,), (2048,), (64,), (64,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 64, 16, 112, 112), (4, 8, 64, 112, 112), (4, 64, 16, 56, 56), (4, 256, 16, 56, 56), (4, 8, 64, 56, 56), (4, 32, 64, 56, 56), (4, 128, 16, 56, 56), (4, 128, 16, 28, 28), (4, 512, 16, 28, 28), (4, 16, 64, 56, 56), (4, 16, 64, 28, 28), (4, 64, 64, 28, 28), (4, 256, 16, 28, 28), (4, 256, 16, 14, 14), (4, 1024, 16, 14, 14), (4, 32, 64, 28, 28), (4, 32, 64, 14, 14), (4, 128, 64, 14, 14), (4, 512, 16, 14, 14), (4, 512, 16, 7, 7), (4, 2048, 16, 7, 7), (4, 64, 64, 14, 14), (4, 64, 64, 7, 7), (4, 256, 64, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool3d': dict(
        name=["max_pool3d"],
        para=dict(
            kernel_size=[(1, 3, 3), (1, 3, 3)],
            stride=[(1, 2, 2), (1, 2, 2)],
            padding=[(0, 1, 1), (0, 1, 1)],
            dilation=[1, 1],
            ceil_mode=[False, False],
            return_indices=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 64, 16, 112, 112), (4, 8, 64, 112, 112)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((4, 64, 16, 56, 56), (4, 16, 16, 56, 56)), ((4, 256, 16, 56, 56), (4, 64, 16, 56, 56)), ((4, 512, 16, 28, 28), (4, 128, 16, 28, 28)), ((4, 1024, 16, 14, 14), (4, 256, 16, 14, 14)), ((4, 256, 1, 1, 1), (4, 2048, 1, 1, 1))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'add_1': dict(
        name=["add"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 256, 16, 56, 56), (4, 32, 64, 56, 56), (4, 512, 16, 28, 28), (4, 64, 64, 28, 28), (4, 1024, 16, 14, 14), (4, 128, 64, 14, 14), (4, 2048, 16, 7, 7), (4, 256, 64, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(4, 256, 16, 56, 56), (4, 32, 64, 56, 56), (4, 512, 16, 28, 28), (4, 64, 64, 28, 28), (4, 1024, 16, 14, 14), (4, 128, 64, 14, 14), (4, 2048, 16, 7, 7), (4, 256, 64, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adaptive_avg_pool3d': dict(
        name=["adaptive_avg_pool3d"],
        para=dict(
            output_size=[(1, 1, 1), (1, 1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 2048, 16, 7, 7), (4, 256, 64, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'dropout': dict(
        name=["dropout"],
        no_output_ref=True,
        para=dict(
            p=[0.5],
            training=[True],
            inplace=[False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 2304, 1, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'linear': dict(
        name=["linear"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 2304)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(174, 2304)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(174,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 174)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(4,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'mul': dict(
        name=["mul"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            other=[1.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mean_1': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_2': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'norm': dict(
        name=["norm"],
        interface=["torch"],
        para=dict(
            p=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3, 1, 7, 7), (64,), (64, 80, 1, 1, 1), (64, 64, 1, 3, 3), (256, 64, 1, 1, 1), (256,), (256, 80, 1, 1, 1), (64, 256, 1, 1, 1), (128, 320, 1, 1, 1), (128,), (128, 128, 1, 3, 3), (512, 128, 1, 1, 1), (512,), (512, 320, 1, 1, 1), (128, 512, 1, 1, 1), (256, 640, 3, 1, 1), (256, 256, 1, 3, 3), (1024, 256, 1, 1, 1), (1024,), (1024, 640, 1, 1, 1), (256, 1024, 3, 1, 1), (512, 1280, 3, 1, 1), (512, 512, 1, 3, 3), (2048, 512, 1, 1, 1), (2048,), (2048, 1280, 1, 1, 1), (512, 2048, 3, 1, 1), (16, 8, 7, 1, 1), (64, 32, 7, 1, 1), (128, 64, 7, 1, 1), (256, 128, 7, 1, 1), (8, 3, 5, 7, 7), (8,), (8, 8, 3, 1, 1), (8, 8, 1, 3, 3), (32, 8, 1, 1, 1), (32,), (8, 32, 3, 1, 1), (16, 32, 3, 1, 1), (16,), (16, 16, 1, 3, 3), (64, 16, 1, 1, 1), (64, 32, 1, 1, 1), (16, 64, 3, 1, 1), (32, 64, 3, 1, 1), (32, 32, 1, 3, 3), (128, 32, 1, 1, 1), (128, 64, 1, 1, 1), (32, 128, 3, 1, 1), (64, 128, 3, 1, 1), (256, 128, 1, 1, 1), (64, 256, 3, 1, 1), (174, 2304), (174,), (324,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'stack': dict(
        name=["stack"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ())],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'add_3': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1e-06],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[40.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reciprocal': dict(
        name=["reciprocal"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_1': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[40.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp': dict(
        name=["clamp"],
        interface=["torch"],
        para=dict(
            max=[1.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_2': dict(
        name=["mul"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3, 1, 7, 7), (64,), (64, 80, 1, 1, 1), (64, 64, 1, 3, 3), (256, 64, 1, 1, 1), (256,), (256, 80, 1, 1, 1), (64, 256, 1, 1, 1), (128, 320, 1, 1, 1), (128,), (128, 128, 1, 3, 3), (512, 128, 1, 1, 1), (512,), (512, 320, 1, 1, 1), (128, 512, 1, 1, 1), (256, 640, 3, 1, 1), (256, 256, 1, 3, 3), (1024, 256, 1, 1, 1), (1024,), (1024, 640, 1, 1, 1), (256, 1024, 3, 1, 1), (512, 1280, 3, 1, 1), (512, 512, 1, 3, 3), (2048, 512, 1, 1, 1), (2048,), (2048, 1280, 1, 1, 1), (512, 2048, 3, 1, 1), (16, 8, 7, 1, 1), (64, 32, 7, 1, 1), (128, 64, 7, 1, 1), (256, 128, 7, 1, 1), (8, 3, 5, 7, 7), (8,), (8, 8, 3, 1, 1), (8, 8, 1, 3, 3), (32, 8, 1, 1, 1), (32,), (8, 32, 3, 1, 1), (16, 32, 3, 1, 1), (16,), (16, 16, 1, 3, 3), (64, 16, 1, 1, 1), (64, 32, 1, 1, 1), (16, 64, 3, 1, 1), (32, 64, 3, 1, 1), (32, 32, 1, 3, 3), (128, 32, 1, 1, 1), (128, 64, 1, 1, 1), (32, 128, 3, 1, 1), (64, 128, 3, 1, 1), (256, 128, 1, 1, 1), (64, 256, 3, 1, 1), (174, 2304), (174,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sgd': dict(
        name=["sgd"],
        interface=["CustomizedTest"],
        para=dict(
            nesterov=[False for i in range(54)],
            lr=[0.005999999999999998 for i in range(54)],
            momentum=[0.9 for i in range(54)],
            weight_decay=[1e-06 for i in range(54)],
            dampening=[0 for i in range(54)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(64, 3, 1, 7, 7), (64,), (64, 80, 1, 1, 1), (64, 64, 1, 3, 3), (256, 64, 1, 1, 1), (256,), (256, 80, 1, 1, 1), (64, 256, 1, 1, 1), (128, 320, 1, 1, 1), (128,), (128, 128, 1, 3, 3), (512, 128, 1, 1, 1), (512,), (512, 320, 1, 1, 1), (128, 512, 1, 1, 1), (256, 640, 3, 1, 1), (256, 256, 1, 3, 3), (1024, 256, 1, 1, 1), (1024,), (1024, 640, 1, 1, 1), (256, 1024, 3, 1, 1), (512, 1280, 3, 1, 1), (512, 512, 1, 3, 3), (2048, 512, 1, 1, 1), (2048,), (2048, 1280, 1, 1, 1), (512, 2048, 3, 1, 1), (16, 8, 7, 1, 1), (64, 32, 7, 1, 1), (128, 64, 7, 1, 1), (256, 128, 7, 1, 1), (8, 3, 5, 7, 7), (8,), (8, 8, 3, 1, 1), (8, 8, 1, 3, 3), (32, 8, 1, 1, 1), (32,), (8, 32, 3, 1, 1), (16, 32, 3, 1, 1), (16,), (16, 16, 1, 3, 3), (64, 16, 1, 1, 1), (64, 32, 1, 1, 1), (16, 64, 3, 1, 1), (32, 64, 3, 1, 1), (32, 32, 1, 3, 3), (128, 32, 1, 1, 1), (128, 64, 1, 1, 1), (32, 128, 3, 1, 1), (64, 128, 3, 1, 1), (256, 128, 1, 1, 1), (64, 256, 3, 1, 1), (174, 2304), (174,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(64, 3, 1, 7, 7), (64,), (64, 80, 1, 1, 1), (64, 64, 1, 3, 3), (256, 64, 1, 1, 1), (256,), (256, 80, 1, 1, 1), (64, 256, 1, 1, 1), (128, 320, 1, 1, 1), (128,), (128, 128, 1, 3, 3), (512, 128, 1, 1, 1), (512,), (512, 320, 1, 1, 1), (128, 512, 1, 1, 1), (256, 640, 3, 1, 1), (256, 256, 1, 3, 3), (1024, 256, 1, 1, 1), (1024,), (1024, 640, 1, 1, 1), (256, 1024, 3, 1, 1), (512, 1280, 3, 1, 1), (512, 512, 1, 3, 3), (2048, 512, 1, 1, 1), (2048,), (2048, 1280, 1, 1, 1), (512, 2048, 3, 1, 1), (16, 8, 7, 1, 1), (64, 32, 7, 1, 1), (128, 64, 7, 1, 1), (256, 128, 7, 1, 1), (8, 3, 5, 7, 7), (8,), (8, 8, 3, 1, 1), (8, 8, 1, 3, 3), (32, 8, 1, 1, 1), (32,), (8, 32, 3, 1, 1), (16, 32, 3, 1, 1), (16,), (16, 16, 1, 3, 3), (64, 16, 1, 1, 1), (64, 32, 1, 1, 1), (16, 64, 3, 1, 1), (32, 64, 3, 1, 1), (32, 32, 1, 3, 3), (128, 32, 1, 1, 1), (128, 64, 1, 1, 1), (32, 128, 3, 1, 1), (64, 128, 3, 1, 1), (256, 128, 1, 1, 1), (64, 256, 3, 1, 1), (174, 2304), (174,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
