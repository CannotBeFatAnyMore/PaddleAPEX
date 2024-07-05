# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle.distributed as dist
import paddle
from .. import config
from ..api_info import API
from .OPTemplate import OPTemplate


class DistHookOp:
    pass


cfg = config.cfg


def recurse(arg, hook_func):
    if isinstance(arg, paddle.Tensor):
        if not arg.stop_gradient:
            return arg.register_hook(hook_func)
    elif isinstance(arg, (list, tuple)):
        out = []
        for item in arg:
            out.append(recurse(item, hook_func))
        return out
    elif isinstance(arg, dict):
        out_dict = {}
        for key, value in arg.items():
            out_dict[key] = recurse(value, hook_func)
            return out_dict
    else:
        return arg


class dist_Template(OPTemplate):
    def forward(self, *args, **kwargs):
        if self.op_name_ not in cfg.Op_count:
            cfg.Op_count[self.op_name_] = 1
            cfg.prefix_op_name_ += "0"
        else:
            cfg.Op_count[self.op_name_] += 1
            cfg.prefix_op_name_ += str(cfg.Op_count[self.op_name_] - 1)
        if cfg.dump_state:
            api_recorder = API(cfg.dump_mode)
            rank = dist.get_rank()
            api_recorder.update_APIInfo(cfg.prefix_op_name_, rank)
            print(*args, **kwargs)
            getattr(DistHookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
            api_recorder.update_real_data(args, kwargs)
            # Dist op has no autograd fn. Evoke hook function implicitly.
            api_recorder.record_dout(None)
        else:
            getattr(DistHookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
        return
