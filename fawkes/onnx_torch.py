"""Run a small feed-forward ONNX graph (conv / batch-norm / relu / pooling / dense) as a torch module,
so gradients flow through it. Enough for insightface's genderage; no new dependency."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OnnxModule(nn.Module):
    SUPPORTED = {"Conv", "BatchNormalization", "Relu", "GlobalAveragePool", "Flatten", "Gemm", "Sub", "Mul",
                 "Add", "Concat"}

    def __init__(self, path):
        super().__init__()
        import onnx
        from onnx import numpy_helper
        model = onnx.load(str(path))
        g = model.graph
        self.nodes = []
        for n in g.node:
            if n.op_type not in self.SUPPORTED:
                raise NotImplementedError(f"ONNX op {n.op_type} is not supported")
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            self.nodes.append((n.op_type, list(n.input), list(n.output), attrs))
        self._names = {}
        for i, init in enumerate(g.initializer):
            buf = torch.from_numpy(np.array(numpy_helper.to_array(init), dtype=np.float32))
            self.register_buffer(f"w{i}", buf)
            self._names[init.name] = f"w{i}"
        self.input_name = [i.name for i in g.input if i.name not in self._names][0]
        self.output_name = g.output[0].name

    def forward(self, x):
        env = {self.input_name: x}
        get = lambda name: env[name] if name in env else getattr(self, self._names[name])
        for op, ins, outs, a in self.nodes:
            if op == "Conv":
                pads = a.get("pads", [0, 0, 0, 0])
                y = F.conv2d(get(ins[0]), get(ins[1]), get(ins[2]) if len(ins) > 2 else None,
                             stride=a.get("strides", [1, 1]), padding=(pads[0], pads[1]),
                             dilation=a.get("dilations", [1, 1]), groups=a.get("group", 1))
            elif op == "BatchNormalization":
                y = F.batch_norm(get(ins[0]), get(ins[3]), get(ins[4]), get(ins[1]), get(ins[2]),
                                 training=False, eps=a.get("epsilon", 1e-5))
            elif op == "Relu":
                y = F.relu(get(ins[0]))
            elif op == "GlobalAveragePool":
                y = get(ins[0]).mean(dim=(2, 3), keepdim=True)
            elif op == "Flatten":
                y = get(ins[0]).flatten(a.get("axis", 1))
            elif op == "Gemm":
                w = get(ins[1])
                w = w.t() if a.get("transB", 0) else w
                y = a.get("alpha", 1.0) * get(ins[0]) @ w
                if len(ins) > 2:
                    y = y + a.get("beta", 1.0) * get(ins[2])
            elif op == "Sub":
                y = get(ins[0]) - get(ins[1])
            elif op == "Mul":
                y = get(ins[0]) * get(ins[1])
            elif op == "Add":
                y = get(ins[0]) + get(ins[1])
            elif op == "Concat":
                y = torch.cat([get(i) for i in ins], dim=a.get("axis", 1))
            env[outs[0]] = y
        return env[self.output_name]
