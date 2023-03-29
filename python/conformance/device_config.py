import copy
from .config import _must_be_the_type, _must_exist, check_dtype_not_nested_list_or_tuple, expand_cfg_by_name


class Skip:
    def __init__(self, value):
        self.value = value


def fix_configs_name(cfgs_dict: dict, base_cfgs_dict: dict):
    for case_k, case_v in cfgs_dict.items():
        if case_k in base_cfgs_dict:
            base_case_v = base_cfgs_dict[case_k]
            if 'name' not in case_v and 'name' in base_case_v:
                case_v['name'] = base_case_v['name']


def check_configs_format(cfgs_dict: dict):
    for case_k, case_v in cfgs_dict.items():
        domain = f"diopi_configs.{case_k}"
        _must_be_the_type(domain, case_v, list, ["dtype", "pytorch"])
        # _must_be_list(case_v, ["dtype", "pytorch"], domain)

        _must_exist(domain, case_v, ['name'])
        _must_be_the_type(domain, case_v, list, ['name', 'arch'])

        if "tensor_para" in case_v.keys():
            _must_be_the_type(domain, case_v, dict, ['tensor_para'])
            if "dtype" in case_v.keys():
                _must_be_the_type(domain, case_v, list, ["dtype"])
                check_dtype_not_nested_list_or_tuple(f"{domain}.dtype",
                                                     case_v["dtype"])

            _must_exist(domain + ".tensor_para", case_v["tensor_para"], ["args"])
            _must_be_the_type(domain + ".tensor_para", case_v["tensor_para"],
                              (list, tuple), ['args'])
            domain_tmp = domain + ".tensor_para.args"

            for arg in case_v["tensor_para"]['args']:
                _must_be_the_type(domain_tmp, arg, (list, tuple),
                                  [i for i in arg.keys() if i != "gen_fn"])
                for arg_k, arg_v in arg.items():
                    if arg_k == "dtype":
                        check_dtype_not_nested_list_or_tuple(
                            f"{domain_tmp}.{arg_k}.dtype", arg_v)

        if "para" in case_v.keys():
            _must_be_the_type(domain, case_v, dict, ['para'])
            dict_obj = case_v["para"]
            _must_be_the_type(domain + ".para", dict_obj, (list, tuple),
                              [i for i in dict_obj.keys() if i != "gen_fn"])

        if "tensor_para" in case_v.keys():
            if "args" in case_v["tensor_para"]:
                args: list = case_v["tensor_para"]["args"]
                domain_tmp0 = domain_tmp + "tensor_para.args"
                for arg in args:
                    _must_be_the_type(domain_tmp0, arg, (list, tuple), ['shape', 'value', 'dtype'])


def expand_tensor_paras_args_by_ins(cfgs_dict):
    '''
    [
        {
            "ins": ['x1', 'x2'],
            "shape": [(2, 3, 16), (4, 32, 7, 7)],
        },
    ]
    ====>
    {
        'x1':{
            "ins": ['x1'],
            "shape": [(2, 3, 16), (4, 32, 7, 7)],
        },
        'x2':{
            "ins": ['x2'],
            "shape": [(2, 3, 16), (4, 32, 7, 7)],
        },
    }
    '''
    for cfg_name in cfgs_dict:
        tensor_para_args = cfgs_dict[cfg_name]["tensor_para"]["args"]
        tmp_tensor_para_args = {}
        for arg in tensor_para_args:
            assert isinstance(arg["ins"], (list, tuple))
            for in_name in arg["ins"]:
                tmp_tensor_para_args[in_name] = copy.deepcopy(arg)
                tmp_tensor_para_args[in_name]["ins"] = [in_name]
        cfgs_dict[cfg_name]["tensor_para"]["args"] = tmp_tensor_para_args


def format_cfg(cases):
    for case_k, case_v in cases.items():
        # set [] for defalut para, tensor_para, para
        if "tensor_para" not in case_v.keys():
            case_v["tensor_para"] = {}
        if "args" not in case_v["tensor_para"].keys():
            case_v["tensor_para"]["args"] = []
        if "para" not in case_v.keys():
            case_v["para"] = {}


def remove_unnecessary_paras(cfgs_dict):
    for case_k, case_v in cfgs_dict.items():
        if "dtype" in case_v.keys():
            case_v["dtype"] = [x.value for x in case_v["dtype"] if isinstance(x, Skip)]
        for para_k, para_v in case_v["para"].items():
            case_v["para"][para_k] = [x.value for x in para_v if isinstance(x, Skip)]
        for arg_k, arg_v in case_v["tensor_para"]["args"].items():
            if "shape" in arg_v:
                arg_v["shape"] = [x.value for x in arg_v["shape"] if isinstance(x, Skip)]
            if "value" in arg_v:
                arg_v["value"] = [x.value for x in arg_v["value"] if isinstance(x, Skip)]
            if "dtype" in arg_v:
                arg_v["dtype"] = [x.value for x in arg_v["dtype"] if isinstance(x, Skip)]


class DeviceConfig(object):
    r"""
    Process device config file
    """

    @staticmethod
    def process_configs(cfgs_dict: dict, base_cfgs_dict: dict):
        fix_configs_name(cfgs_dict, base_cfgs_dict)
        check_configs_format(cfgs_dict)
        cfgs_dict = expand_cfg_by_name(cfgs_dict, 'name')
        format_cfg(cfgs_dict)
        expand_tensor_paras_args_by_ins(cfgs_dict)
        remove_unnecessary_paras(cfgs_dict)
        return cfgs_dict
