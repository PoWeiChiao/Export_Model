import numpy as np

import torch
import torch.onnx
import torchvision.models as models

import onnx
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = resnet(x)
    torch.onnx.export(resnet,
                    x,
                    'resnet.onnx',
                    export_params=True,
                    opset_version=10,
                    do_constant_folding=True,
                    input_names=['modelInput'],
                    output_names=['modelOutput'],
                    dynamic_axes={'modelInput' : {0 : 'batch_size'}, 'modelOutput' : {0 : 'batch_size'}})

    onnx_model = onnx.load("resnet.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("resnet.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
                    
if __name__ == '__main__':
    main()