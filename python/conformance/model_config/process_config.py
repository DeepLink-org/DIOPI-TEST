import cv_config
import seg_config
import os

unary_op = {'input': 'tensor', 'inplace': 'para/key'}
func_para = dict(
    relu=unary_op,
    floor=unary_op,
    neg=unary_op,
    reciprocal=unary_op,
    conv2d={"input": "tensor/grad", "weight": "tensor/grad", "bias": "tensor/none/grad",
            "stride": "para", "padding": "para", "dilation": "para", "groups": "para"},
    batch_norm={"input": "tensor/grad", "running_mean": "tensor", "running_var": "tensor", "weight": "tensor/none/grad",
                "bias": "tensor/none/grad", "training": "para", "momentum": "para", "eps": "para"},
    max_pool2d={"input": "tensor/grad", "kernel_size": "para", "stride": "para", "padding": "para",
                "dilation": "para", "ceil_mode": "para", "return_indices": "para"},
    adaptive_avg_pool2d={"input": "tensor/grad", "output_size": "para"},
    linear={"input": "tensor/grad", "weight": "tensor/grad", "bias": "tensor/none/grad"},
    cross_entropy={"input": "tensor/grad", "target": "tensor", "weight": "tensor/none/grad", "size_average": "para/key",
                   "ignore_index/key": "para", "reduce": "para/key", "reduction": "para/key", "label_smoothing": "para/key"},
    add={"input": "tensor", "other": "tensor/scalar", "alpha": "para/key"},
    sum={"input": "tensor", "dim": "para/key", "dtype": "para/key"},
    mean={"input": "tensor", "dim": "para/key", "dtype": "para/key"},
    mul={"input": "tensor", "other": "tensor/scalar"},
    div={"input": "tensor", "other": "tensor/scalar", "rounding_mode": "para/key"},
    randperm={"n": "para", "dtype": "para/key"},
    sgd={'param", "param_grad': "tensor", "buf": "tensor", "nesterov": "para/key", "lr": "para/key", "momentum": "para/key",
         "weight_decay": "para/key", "dampening": "para/key"},
    cat={'tensors': "tensorlist", 'dim': "para/key"},
    avg_pool2d={"input": "tensor/grad", "kernel_size": "para", "stride": "para/key", "padding": "para/key", "ceil_mode": "para/key",
                "count_include_pad": "para/key", "divisor_override": "para/key"},
    sigmoid={'input': "tensor/grad"},
    hardtanh={'input': "tensor/grad", 'min_val': "para/key", "max_val": "para/key", "inplace": "para/key"},
    linspace={'start': "para", 'end': "para", "steps": 'para', 'dtype': 'para/key'},
    pad={'input': "tensor", 'pad': 'para', 'mode': "para/key", 'value': 'para'},
    transpose={'input': 'tensor', 'dim0': 'para', 'dim1': 'para'},
    dropout={'input': 'tensor', 'p': 'para/key', 'training': 'para/key', 'inplace': 'para/key'},
    arange={'start': "para/key", "end": "para", "step": "para/key", 'dtype': 'para/key'},
    one_hot={'input': 'tensor', 'num_classes': 'para/key'},
    layer_norm={'input': 'tensor/grad', 'normalized_shape': 'para/key', 'weight': 'tensor/none/grad', 'bias': 'tensor/none/grad', 'eps': 'para/key'},
    permute={'input': 'tensor', 'dims': 'para/key'},
    matmul={'input': 'tensor', 'other': 'tensor'},
    softmax={'input': 'tensor', 'dim': 'para', 'dtype': 'para/key'},
    gelu={'input': 'tensor/grad', 'approximate': 'para/key'},
    roll={'input': 'tensor', 'shifts': 'para/key', 'dims': 'para/key'},
    sub={"input": "tensor", "other": "tensor/scalar", "alpha": "para/key"},
    ne={"input": "tensor", "other": "tensor/scalar", "inplace": 'para/key'},
    eq={"input": "tensor", "other": "tensor/scalar", "inplace": 'para/key'},
    masked_fill={'input': 'tensor', 'mask': 'tensor', 'value': 'tensor/scalar', 'inplace': 'para/key'},
    log_softmax={'input': 'tensor', 'dim': 'para/key', 'dtype': 'para/key'},
    unfold={'input': 'tensor/grad', 'dimension': 'para', 'size': 'para', 'step': 'para'},
    im2col={'input': 'tensor', 'kernel_size': 'para', 'dilation': 'para/key', 'padding': 'para/key', 'stride': 'para/key'},
    norm={'input': 'tensor', 'p': 'para', 'dim': 'para/key', 'keepdim': 'para/key', 'dtype': 'para/key'},
    stack={'tensors': "tensorlist", 'dim': "para/key"},
    clamp={'input': 'tensor', 'min': 'tensor/scalar/key', 'max': 'tensor/scalar/key', 'inplace': 'para/key'},
    addcmul={'input': 'tensor', 'tensor1': 'tensor', 'tensor2': 'tensor', 'value': 'para/key'},
    sqrt={'input': "tensor"},
    addcdiv={'input': 'tensor', 'tensor1': 'tensor', 'tensor2': 'tensor', 'value': 'para/key'},
    expand={'input': 'tensor', 'size': 'para'},
    tanh={'input': "tensor/grad"},
    uniform={'input': 'tensor', 'start': 'para/key', 'end': 'para/key'},
    adamw={'param", "param_grad': "tensor", 'exp_avg", "exp_avg_sq", "max_exp_avg_sq': "tensor", 'step': 'para',
           "amsgrad": "para/key", "beta1": "para/key", "beta2": "para/key", "lr": "para/key", "weight_decay": "para/key", "eps": "para/key"},
    interpolate={'input': 'tensor', 'size': 'para/key', 'mode': 'para/key', 'align_corners': 'para/key'},
    topk={'input':'tensor','k':'para','dim':'para/key','largest':'para/key','sorted':'para/key'},

)
convert_name = {'iadd': "add", 'radd': "add", 'add_': "add", 'rmul': 'mul', 'truediv': 'div', 'rtruediv': 'div',
                'mul_': 'mul','div_':'div', 'addcmul_': 'addcmul', 'addcdiv_': 'addcdiv', 'uniform_': 'uniform', 'rand': 'uniform','relu_':'relu'}
inplace_tag = ['iadd', 'mul_', 'div_']
interface_tag = {"sgd": "CustomizedTest", "adamw": "CustomizedTest", 'im2col': 'CustomizedTest'}
no_output_ref = ['randperm', 'uniform', 'dropout']
saved_args = {"sigmoid": "0", 'softmax': '0', 'log_softmax': '0', 'tanh': '0'}

tensor_vide = "                    "
para_vide = "            "
key_vide = "        "
seq_name = ['cat', 'stack']


def toDtype(dtype, tensor_para):
    if dtype == 'torch.cuda.FloatTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.float32],\n')
        tensor_para.append(tensor_vide + '"gen_fn": Genfunc.randn,\n')
    elif dtype == 'torch.cuda.DoubleTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.float32],\n')
        tensor_para.append(tensor_vide + '"gen_fn": Genfunc.randn,\n')
    elif dtype == 'torch.cuda.LongTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.int64],\n')
        tensor_para.append(tensor_vide + '"gen_fn": Genfunc.randint,\n')
    elif dtype == 'torch.cuda.BoolTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.bool],\n')
        tensor_para.append(tensor_vide + '"gen_fn": Genfunc.mask,\n')
    elif isinstance(dtype, list):
        dtype_list = [ele.replace("torch", "Dtype") for ele in dtype]
        dtype_list = list(set(dtype_list))
        tensor_para.append(tensor_vide + '"dtype": ' + str(dtype_list).replace("'", "") + ',\n')
        tensor_para.append(tensor_vide + '"gen_fn": Genfunc.randn,\n')


def gen_config_code(config, file_name):
    content = config
    names = {}

    os.system(f"rm -f {file_name}.py")
    with open(f'{file_name}.py', 'a') as f:
        f.write("from ..config import Genfunc\n")
        f.write("from ..dtype import Dtype\n\n")
        f.write(file_name + " = {\n")

    for ele in content:
        name = ele[0]
        if name == 'unfold' and ele[1] == 'torch.nn.functional':
            name = 'im2col'
        if name in convert_name.keys():
            name = convert_name[name]
        para = []
        tensor_para = []
        if name not in names.keys():
            config = ["    '" + name + "': dict(\n"]
            names.update({name: 0})
        else:
            names[name] += 1
            config = ["    '" + name + "_" + str(names[name]) + "': dict(\n"]

        config.append(key_vide + 'name=["' + name + '"],\n')
        para_list = ele[3]
        kpara_list = ele[4]
        type_list = ele[2]
        idx = 0
        type_idx = 0
        if name not in func_para.keys():
            print(f"%%%%%%% miss definition for {name} while generate {file_name}.py %%%%%%%%%%\n")
            continue
        if ele[0] in inplace_tag:
            config.append(key_vide + 'is_inplace=[True],\n')

        if name in no_output_ref:
            config.append(key_vide + 'no_output_ref=True,\n')
        elif name in interface_tag.keys():
            config.append(key_vide + 'interface=["' + interface_tag[name] + '"],\n')
        elif ele[1] != 'torch.nn.functional':
            config.append(key_vide + 'interface=["' + ele[1] + '"],\n')

        if name in saved_args.keys():
            config.append(key_vide + 'saved_args=dict(output=' + saved_args[name] + '),\n')

        for k, v in func_para[name].items():
            if idx >= len(para_list) + len(kpara_list):
                break
            is_para = True
            if "tensor" in v:
                if idx >= len(para_list):
                    if "scalar/key" not in v:
                        assert "none" in v, "tensor can not be None"
                        continue
                elif isinstance(para_list[idx][0], tuple):
                    is_para = False

                if not is_para:
                    tensor_para.append(para_vide + "    {\n" + tensor_vide + '"ins": ["' + str(k) + '"],\n')
                    if "grad" in v and type_idx < len(type_list):
                        tensor_para.append(tensor_vide + '"requires_grad": [True],\n')
                    if name in ['sgd', 'adamw']:
                        length = len(para_list[idx])
                        if length == 1:
                            para_list[idx] = para_list[idx][0]
                        else:
                            tmp_list = []
                            for i in range(length):
                                tmp_list.append(para_list[idx][i][0])
                            para_list[idx] = tmp_list

                    tensor_para.append(tensor_vide + '"shape": ' + str(para_list[idx]) + ",\n")
                    if ele[0] == 'rand':
                        toDtype(kpara_list['dtype'], tensor_para)
                    elif type_idx < len(type_list):
                        toDtype(type_list[type_idx], tensor_para)
                        type_idx += 1
                    tensor_para.append(para_vide + "    },\n")

            if is_para:
                if "key" in v and k in kpara_list.keys():
                    if name in ['sgd', 'adamw'] and not isinstance(kpara_list[k], list):
                        para.append(para_vide + str(k) + "=[" + str(kpara_list[k]) + f" for i in range({len(para_list[0])})],\n")
                    else:
                        if name == 'interpolate' and k == 'size':
                            kpara_list[k] = [ tuple(e) for e in kpara_list[k]]
                        if not isinstance(kpara_list[k], list):
                            kpara_list[k] = [kpara_list[k]]
                        para.append(para_vide + str(k) + "=" + str(kpara_list[k]) + ",\n")
                elif idx < len(para_list):
                    if name in ['permute', 'expand'] and idx + 1 < len(para_list):
                        dims_list = []
                        for i in range(len(para_list[idx])):
                            dims = [para_list[j][i] for j in range(idx, len(para_list))]
                            dims_list.append(tuple(dims))
                        para_list[idx] = dims_list

                    if name == 'adamw' and k == 'step':
                        idx += 3
                        step_list = [e[0] for e in para_list[idx]]
                        para_list[idx] = step_list
                    para.append(para_vide + str(k) + "=" + str(para_list[idx]) + ",\n")
                else:
                    continue
            idx += 1

        if para:
            config.append(key_vide + "para=dict(\n")
            for e in para:
                config.append(e)
            config.append(key_vide + "),\n")
        if tensor_para:
            config.append(key_vide + "tensor_para=dict(\n")
            config.append(para_vide + "args=[\n")
            for e in tensor_para:
                config.append(e)
            config.append(para_vide + "],\n")
            if name in seq_name:
                config.append(para_vide + "seq_name='tensors',\n")
            config.append(key_vide + "),\n")
        config.append("    ),\n")
        config.append("\n")

        with open(f'{file_name}.py', 'a') as f:
            for row in config:
                f.write(row)
    with open(f'{file_name}.py', 'a') as f:
        f.write("}\n")


if __name__ == '__main__':
    config_dict = {##"resnet50_config": cv_config.resnet50_8xb32_in1k_config,
                #    'resnet101_config': cv_config.resnet101_8xb32_in1k_config,
                #    'densenet_config': cv_config.densenet121_4xb256_in1k_config,
                #    'seresnet50_config': cv_config.seresnet50_8xb32_in1k_config,
                #    'efficientnet_config': cv_config.efficientnet_b2_8xb32_in1k_config,
                #    "mobilenet_v2_config": cv_config.mobilenet_v2_8xb32_in1k_config,
                #    "repvgg_config": cv_config.repvgg_A0_4xb64_coslr_120e_in1k_config,
                #    "shufflenet_v2_config": cv_config.shufflenet_v2_1x_16xb64_in1k_config,
                #    "swin_transformer_config": cv_config.swin_base_16xb64_in1k_config,
                #    "vit_config": cv_config.vit_base_p16_pt_64xb64_in1k_224_config,
                #    "vgg16_config": cv_config.vgg16_8xb32_in1k_config,
                   "unet_config":seg_config.fcn_unet_s5_d16_4x4_512x1024_160k_cityscapes_config,
                    "fcn_config":seg_config.fcn_d6_r50_d16_512x1024_40k_cityscapes_config,
                    "deeplabv3_config":seg_config.deeplabv3_r50_d8_512x1024_40k_cityscapes_config,
                    "deeplabv3plus_config":seg_config.deeplabv3plus_r50_d8_512x1024_40k_cityscapes_config,
                    "pspnet_config":seg_config.pspnet_r50_d8_512x1024_40k_cityscapes_config,
                    "upernet_config":seg_config.upernet_r50_512x1024_40k_cityscapes_config}
    for k, v in config_dict.items():
        gen_config_code(v, k)
