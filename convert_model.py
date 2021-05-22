# TODO: 최종 변환 후에 기존 pytorch 출력텐서 값과 tensorflow 출력 텐서 값을 확인!!

import io
import numpy as np

from torch import nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.onnx
from train import main

import onnx
from onnx_tf.backend import prepare
from train import set_hyperparameters
from adversarial_training import main


def torch2onnx():
    # PYTORCH 모델(.pt)을 ONNX으로 변환
    # https://tutorials.pytorch.kr/advanced/super_resolution_with_onnxruntime.html

    # 모델을 미리 학습된 가중치로 초기화
    torch_model, x = main()

    # 미리 학습된 가중치를 읽어옴
    # eval()모드를 적용할 수 있어야 하므로, torch.save(model, PATH + 'model.pt')전체 모델을 저장한 pt파일이여야 함.
    model_url = 'C:/Users/mmclab1/Desktop/fakecheck/pytorch_model_adv_epoch20.pt'
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # 모델을 추론 모드로 전환
    torch_model.eval()

    # 모델에 대한 입력값; export 함수가 모델을 실행하기 때문에, 직접 텐서를 입력값으로 넘겨주어야 함.
    # model, x = main()

    # 모델 변환
    # RuntimeError: ONNX export failed: Couldn't export Python operator SwishImplementation
    # torch_model.set_swish(memory_efficient=False)

    # 모델을 실행하여 어떤 연산자들이 출력값을 계산하는데 사용되었는지를 기록
    torch.onnx.export(torch_model,               # 실행될 모델
                      x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      'C:/Users/mmclab1/Desktop/fakecheck/onnx_model_adv.onnx',         # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                      input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                      output_names = ['output'], # 모델의 출력값을 가리키는 이름
                      dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                    'output' : {0 : 'batch_size'}})


def onnx2tf():
    # https://ichi.pro/ko/pytorchleul-tensorflow-litelo-byeonhwanhaneun-naui-yeojeong-14449401839341
    # https://github.com/onnx/onnx-tensorflow

    TF_PATH = "C:/Users/mmclab1/Desktop/fakecheck/tensorflow_model_adv"
    ONNX_PATH = "C:/Users/mmclab1/Desktop/fakecheck/onnx_model_adv.onnx"

    onnx_model = onnx.load(ONNX_PATH)  # load onnx model
    print('success loading onnx file')

    tf_rep = prepare(onnx_model)  # creating TensorflowRep object
    '''
    Process finished with exit code -1073741819 (0xC0000005)
    Error code 0xc0000005 means "access violation"
    -> Please check your MPI code for memory/stack/heap access issues. And make sure any array access has valid index.
    
    ==> 해결) 읽어들인 학습된 가중치(.pt)는 torch.save(model.state_dict(), PATH) 모델 객체의 state_dict가 저장되었기 때문.
    model.load_state_dict(best_model_wts) 전체모델 저장으로 바꿔서 수행
    '''
    print('success creating tensorflowRep')


    tf_rep.export_graph(TF_PATH)    # export the model
    print('success converting onnx to tensorflow')


def tf2lite():
    # https://www.tensorflow.org/lite/convert?hl=ko

    # Not creating XLA devices, tf_xla_enable_xla_devices not set
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf

    # Convert the model
    saved_model_dir = "C:/Users/mmclab1/Desktop/fakecheck/tensorflow_model_adv"
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Save the model.
    with open('tfLite_model.tflite', 'wb') as f:
        f.write(tflite_model)



if __name__ == '__main__':
    # torch2onnx()      # pytorch(.pt) --> onnx
    # onnx2tf()         # onnx --> tensorflow(.pb)
    tf2lite()           # tensorflow(.pb) --> tensorflow lite(.tflite)