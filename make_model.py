"""Builds difference_matte.onnx: CIELAB distance between two sRGB images -> soft alpha.
Inputs : canvas, background  float32 [1,3,H,W] RGB 0..255
         tolerance, softness, lightness_weight  float32 [1]
Output : alpha float32 [1,1,H,W]
"""
import numpy as np, onnx
from onnx import helper, TensorProto as T, numpy_helper as nh

nodes, inits, n = [], [], [0]
def const(val, shape=None, dtype=np.float32):
    n[0] += 1; name = f"c{n[0]}"
    inits.append(nh.from_array((np.array(val, dtype=dtype).reshape(shape) if shape else np.array(val, dtype=dtype)), name)); return name
def op(kind, ins, **kw):
    n[0] += 1; out = f"t{n[0]}"
    nodes.append(helper.make_node(kind, ins, [out], **kw)); return out

W_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], np.float32)
W_XYZ[0] /= 0.95047; W_XYZ[2] /= 1.08883
W_LAB = np.array([[0, 116, 0], [500, -500, 0], [0, 200, -200]], np.float32)
w_xyz = const(W_XYZ, (3, 3, 1, 1)); w_lab = const(W_LAB, (3, 3, 1, 1)); b_lab = const([-16, 0, 0], (3,))
c_255, c_04045, c_1292 = const(255.), const(0.04045), const(12.92)
c_0055, c_1055, c_24 = const(0.055), const(1.055), const(2.4)
c_eps, c_third, c_7787, c_16116 = const(0.008856), const(1 / 3), const(7.787), const(16 / 116)

def to_lab(x255):
    x = op("Div", [x255, c_255])
    lin = op("Where", [op("Less", [x, c_04045]), op("Div", [x, c_1292]),
                       op("Pow", [op("Div", [op("Add", [x, c_0055]), c_1055]), c_24])])
    xyz = op("Conv", [lin, w_xyz])
    f = op("Where", [op("Greater", [xyz, c_eps]), op("Pow", [xyz, c_third]),
                     op("Add", [op("Mul", [xyz, c_7787]), c_16116])])
    return op("Conv", [f, w_lab, b_lab])

lab_c, lab_b = to_lab("canvas"), to_lab("background")
diff = op("Sub", [lab_c, lab_b])
wvec = op("Reshape", [op("Concat", ["lightness_weight", const([1, 1])], axis=0), const([1, 3, 1, 1], dtype=np.int64)])
diff = op("Mul", [diff, wvec])
dist = op("Sqrt", [op("ReduceSum", [op("Mul", [diff, diff]), const([1], dtype=np.int64)], keepdims=1)])
soft = op("Max", ["softness", const(0.001)])
t = op("Clip", [op("Div", [op("Sub", [dist, "tolerance"]), soft]), const(0.), const(1.)])
alpha = op("Mul", [op("Mul", [t, t]), op("Sub", [const(3.), op("Mul", [const(2.), t])])])
nodes.append(helper.make_node("Identity", [alpha], ["alpha"]))

def vi(name, shape): return helper.make_tensor_value_info(name, T.FLOAT, shape)
g = helper.make_graph(nodes, "difference_matte",
    [vi("canvas", [1, 3, "H", "W"]), vi("background", [1, 3, "H", "W"]),
     vi("tolerance", [1]), vi("softness", [1]), vi("lightness_weight", [1])],
    [vi("alpha", [1, 1, "H", "W"])], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
onnx.checker.check_model(m); onnx.save(m, "difference_matte.onnx")

# ---- verify against a numpy reference of the C# CPU path
import onnxruntime as ort
def srgb(c):
    c = c / 255.; return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
def lab(rgb):
    l = srgb(rgb) @ W_XYZ.T
    f = np.where(l > .008856, np.cbrt(l), 7.787 * l + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)
def ref(c, b, tol, soft, lw):
    d = lab(c) - lab(b); d[..., 0] *= lw
    d = np.sqrt((d ** 2).sum(-1))
    t = np.clip((d - tol) / max(soft, .001), 0, 1); return t * t * (3 - 2 * t)

rng = np.random.default_rng(1)
b = rng.integers(0, 256, (37, 53, 3)).astype(np.float32)
c = np.clip(b + rng.normal(0, 25, b.shape) * (rng.random(b.shape[:2] + (1,)) > .5), 0, 255).astype(np.float32)
s = ort.InferenceSession("difference_matte.onnx", providers=["CPUExecutionProvider"])
for tol, soft, lw in [(10, 8, .6), (5, 0, 1.), (20, 30, 0.)]:
    out = s.run(None, {"canvas": c.transpose(2, 0, 1)[None], "background": b.transpose(2, 0, 1)[None],
        "tolerance": np.float32([tol]), "softness": np.float32([soft]), "lightness_weight": np.float32([lw])})[0][0, 0]
    print(tol, soft, lw, "max abs err:", np.abs(out - ref(c, b, tol, soft, lw)).max(), "shape", out.shape)
