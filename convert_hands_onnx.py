from pathlib import Path
import onnx
from onnx import helper
from rknn.api import RKNN

OUT_DIR = Path(__file__).resolve().parent / "models" / "mediapipe_rknn"
DATASET_TXT = "/tmp/rknn_calib/dataset.txt"

# (onnx_path, rknn_path, img_size, mean_vals, std_vals)
models = [
    (
        OUT_DIR / "extracted" / "hand_detector.onnx",
        OUT_DIR / "hand_detector.rknn",
        192,
        [[127.5, 127.5, 127.5]],
        [[127.5, 127.5, 127.5]],
    ),
    (
        OUT_DIR / "extracted" / "hand_landmarks_detector.onnx",
        OUT_DIR / "hand_landmarks_detector.rknn",
        224,
        [[0, 0, 0]],
        [[255, 255, 255]],
    ),
]


def make_nchw(src_onnx: Path, dst_onnx: Path, size: int):
    model = onnx.load(str(src_onnx))
    graph = model.graph
    orig_in = graph.input[0]
    orig_name = orig_in.name

    nhwc_name = orig_name + "_nhwc"
    new_in = helper.make_tensor_value_info(orig_name, onnx.TensorProto.FLOAT, [1, 3, size, size])
    transpose_node = helper.make_node(
        "Transpose", inputs=[orig_name], outputs=[nhwc_name], perm=[0, 2, 3, 1]
    )

    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == orig_name:
                node.input[i] = nhwc_name

    graph.input.remove(orig_in)
    graph.input.insert(0, new_in)
    graph.node.insert(0, transpose_node)
    onnx.save(model, str(dst_onnx))


for onnx_path, rknn_path, img_size, mean_vals, std_vals in models:
    print(f"\n=== Converting {onnx_path.name} -> {rknn_path.name} (mean={mean_vals[0][0]}, std={std_vals[0][0]}) ===")
    nchw_onnx = onnx_path.with_name(onnx_path.stem + "_nchw.onnx")
    make_nchw(onnx_path, nchw_onnx, img_size)

    rknn = RKNN(verbose=False)
    try:
        assert rknn.config(mean_values=mean_vals, std_values=std_vals, target_platform="rv1106") == 0
        assert rknn.load_onnx(model=str(nchw_onnx)) == 0
        assert rknn.build(do_quantization=True, dataset=str(DATASET_TXT)) == 0
        assert rknn.export_rknn(str(rknn_path)) == 0
        print(f"SUCCESS: Created {rknn_path}")
    finally:
        rknn.release()
