import argparse
import os
import time
import paddle
import copy
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath("../"))
from utils import (
    print_info_log,
    check_grad_list,
    gen_api_params,
    api_json_read,
    rand_like,
    print_warn_log,
)

type_map = {
    "FP16": paddle.float16,
    "FP32": paddle.float32,
    "BF16": paddle.bfloat16,
}


current_time = time.strftime("%Y%m%d%H%M%S")

tqdm_params = {
    "smoothing": 0,  # 平滑进度条的预计剩余时间，取值范围0到1
    "desc": "Processing",  # 进度条前的描述文字
    "leave": True,  # 迭代完成后保留进度条的显示
    "ncols": 75,  # 进度条的固定宽度
    "mininterval": 0.1,  # 更新进度条的最小间隔秒数
    "maxinterval": 1.0,  # 更新进度条的最大间隔秒数
    "miniters": 1,  # 更新进度条之间的最小迭代次数
    "ascii": None,  # 根据环境自动使用ASCII或Unicode字符
    "unit": "it",  # 迭代单位
    "unit_scale": True,  # 自动根据单位缩放
    "dynamic_ncols": True,  # 动态调整进度条宽度以适应控制台
    "bar_format": "{l_bar}{bar}| {n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",  # 自定义进度条输出
}


def recursive_arg_to_device(arg_in, enforce_dtype=None):
    if isinstance(arg_in, (list, tuple)):
        return type(arg_in)(recursive_arg_to_device(arg) for arg in arg_in)
    elif isinstance(arg_in, paddle.Tensor):
        arg_in = arg_in.cuda()
        if enforce_dtype and arg_in.dtype.name in ["BF16", "FP16", "FP32"]:
            arg_in = arg_in.cast(enforce_dtype)
        return arg_in
    else:
        return arg_in


def ut_case_parsing(forward_content, cfg, out_path):
    print_info_log("start UT save")
    multi_dtype_ut = cfg.multi_dtype_ut.split(",") if cfg.multi_dtype_ut else []
    multi_dtype_ut = [type_map[item] for item in multi_dtype_ut]
    fwd_output_dir = os.path.abspath(os.path.join(out_path, "output"))
    bwd_output_dir = os.path.abspath(os.path.join(out_path, "output_backward"))
    os.makedirs(fwd_output_dir, exist_ok=True)
    os.makedirs(bwd_output_dir, exist_ok=True)
    filename = os.path.join(out_path, "./warning_log.txt")
    for i, (api_call_name, api_info_dict) in enumerate(
        tqdm(forward_content.items(), **tqdm_params)
    ):
        if len(multi_dtype_ut) > 0:
            for enforce_dtype in multi_dtype_ut:
                print(api_call_name + "*" + enforce_dtype.name)
                api_info_dict_copy = copy.deepcopy(api_info_dict)
                fwd_res, bp_res = run_api_case(
                    api_call_name, api_info_dict_copy, filename, enforce_dtype
                )
                if enforce_dtype:
                    save_name = api_call_name + "*" + enforce_dtype.name
                fwd_output_path = os.path.join(fwd_output_dir, save_name)
                bwd_output_path = os.path.join(bwd_output_dir, save_name)
                if not isinstance(fwd_res, type(None)):
                    paddle.save(fwd_res, fwd_output_path)
                if not isinstance(bp_res, type(None)):
                    paddle.save(bp_res, bwd_output_path)
                print("*" * 100)
        else:
            print(api_call_name)
            fwd_res, bp_res = run_api_case(api_call_name, api_info_dict, filename)
            fwd_output_path = os.path.join(fwd_output_dir, api_call_name)
            bwd_output_path = os.path.join(bwd_output_dir, api_call_name)
            if not isinstance(fwd_res, type(None)):
                paddle.save(fwd_res, fwd_output_path)
            if not isinstance(bp_res, type(None)):
                paddle.save(bp_res, bwd_output_path)
            print("*" * 100)


def run_api_case(api_call_name, api_info_dict, warning_log_pth, enforce_dtype=None):
    Warning_list = []
    api_call_stack = api_call_name.rsplit("*")[0]
    api_name = api_call_stack.rsplit(".")[-1]
    args, kwargs, need_backward = gen_api_params(api_info_dict)
    if api_name == "scatter_nd":
        return None, None

    ##################################################################
    ##      RUN FORWARD
    ##################################################################
    try:
        device_args = recursive_arg_to_device(args, enforce_dtype)
        device_kwargs = {
            key: recursive_arg_to_device(value, enforce_dtype)
            for key, value in kwargs.items()
        }
        device_out = eval(api_call_stack)(*device_args, **device_kwargs)

    except Exception as err:
        api_name = api_call_name.split("*")[0]
        msg = f"Run API {api_name} Forward Error: %s" % str(err)
        print_warn_log(msg)
        Warning_list.append(msg)
        File = open(warning_log_pth, "a")
        for item in Warning_list:
            File.write(item + "\n")
        File.close()
        return None, None

    ##################################################################
    ##      RUN BACKWARD
    ##################################################################
    if need_backward:
        try:
            device_grad_out = []
            dout = rand_like(device_out)

            dout = recursive_arg_to_device(dout)
            paddle.autograd.backward([device_out], [dout])
            for arg in device_args:
                if isinstance(arg, paddle.Tensor):
                    device_grad_out.append(arg.grad)
            for k, v in device_kwargs.items():
                if isinstance(v, paddle.Tensor):
                    device_grad_out.append(v.grad)
            device_grad_out = check_grad_list(device_grad_out)
        except Exception as err:
            api_name = api_call_name.split("*")[0]
            msg = f"Run API {api_name} backward Error: %s" % str(err)
            print_warn_log(msg)
            Warning_list.append(msg)
            return device_out, None
    else:
        msg = f"{api_call_name} has no tensor required grad, SKIP Backward"
        print_warn_log(msg)
        Warning_list.append(msg)
        return device_out, None

    File = open(warning_log_pth, "a")
    for item in Warning_list:
        File.write(item + "\n")
    File.close()
    return device_out, device_grad_out


def arg_parser(parser):
    parser.add_argument(
        "-json",
        "--json",
        dest="json_path",
        default="",
        type=str,
        help="Dump json file path",
        required=True,
    )
    parser.add_argument(
        "-out",
        "--dump_path",
        dest="out_path",
        default="./root/paddlejob/workspace/PaddleAPEX_dump/",
        type=str,
        help="<optional> The ut task result out path.",
        required=False,
    )
    parser.add_argument(
        "-enforce-dtype",
        "--dtype",
        dest="multi_dtype_ut",
        default="FP32,FP16,BF16",
        type=str,
        help="",
        required=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg_parser(parser)
    cfg = parser.parse_args()
    forward_content = api_json_read(cfg.json_path)
    out_path = os.path.realpath(cfg.out_path) if cfg.out_path else "./"
    ut_case_parsing(forward_content, cfg, out_path)
    print_info_log("UT save completed")