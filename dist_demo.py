from paddle.distributed import fleet
import paddle.distributed as dist
from paddleapex import Tracer
from paddle import framework

apex = Tracer()
import paddle
strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 4,
    "pp_degree": 2,
    "sharding_degree": 1,
}
fleet.init(is_collective=True, strategy=strategy)
rank = dist.get_rank()
group = dist.get_group()
group = (
    paddle.distributed.collective._get_global_group()
    if group is None
    else group
)
print(group)
apex.start()
x = paddle.zeros([8])
x[rank] = rank
group.process_group.all_reduce(tensor = x, op = framework.core.ReduceOp.SUM, sync_op = True)
paddle.distributed.all_reduce(x)
apex.stop()