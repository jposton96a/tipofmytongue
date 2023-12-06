# requirements: pip install optimum[onnxruntime]
import numpy as np


def convert_model_to_onnx(model_id, save=False, save_path=None):
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
    if save:
        model.save_pretrained(save_path)

    return model


def quantize_model(model_id, save_path):
    # repo: https://github.com/philschmid/optimum-transformers-optimizations/tree/master
    # made by an official HF employee
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    # create ORTQuantizer and define quantization configuration
    model = convert_model_to_onnx(model_id)
    dynamic_quantizer = ORTQuantizer.from_pretrained(model)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    # apply the quantization configuration to the model
    model_quantized_path = dynamic_quantizer.quantize(
        save_dir=save_path,
        quantization_config=dqconfig
    )
    return model_quantized_path


def create_warmup_file(save_path, np_array=None, string=False, img_path=None):
    if hasattr(np_array, 'shape'):
        np_array.tofile(save_path)
    elif isinstance(img_path, str):
        import cv2
        img = cv2.imread(img_path)
        img.tofile(save_path)
    elif isinstance(string, str):
        from tritonclient.utils import serialize_byte_tensor
        serialized = serialize_byte_tensor(np.array([string.encode("utf-8")], dtype=object))
        with open(save_path, "wb") as f:
            f.write(serialized.item())
    else:
        print("Invalid input. Input a numpy array, string, or image path")


if __name__ == "__main__":
    # create_warmup_file(
    #     "../../triton/postprocess/warmup/last_hidden_state",
    #     np.random.normal(0, 0.1, size=(1, 3, 1024))
    # )
    # create_warmup_file(
    #     "../../triton/postprocess/warmup/attention_mask",
    #     np.array([[1, 1, 1]])
    # )
    quantize_model("thenlper/gte-large", "./model")
