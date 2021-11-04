"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
import onnx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--onnx2pb', action='store_true', default=False, help='export onnx to pb')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    model.eval()
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
        if isinstance(m, models.common.ShuffleV2Block):#shufflenet block nn.SiLU
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
            for i in range(len(m.branch2)):
                if isinstance(m.branch2[i], nn.SiLU):
                    m.branch2[i] = SiLU()
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    model.fuse()  # only for ONNX
    torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['data'],
                      output_names=['stride_' + str(int(x)) for x in model.stride])

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % f)
    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
    
    # PB export
    if opt.onnx2pb:
        print('download the newest onnx_tf by https://github.com/onnx/onnx-tensorflow/tree/master/onnx_tf')
        from onnx_tf.backend import prepare
        import tensorflow as tf

        outpb = f.replace('.onnx', '.pb')  # filename
        # strict=True maybe leads to KeyError: 'pyfunc_0', check: https://github.com/onnx/onnx-tensorflow/issues/167
        tf_rep = prepare(onnx_model, strict=False)  # prepare tf representation
        tf_rep.export_graph(outpb)  # export the model

        out_onnx = tf_rep.run(img) # onnx output

        # check pb
        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with open(outpb, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                input_x = sess.graph.get_tensor_by_name(input_names[0]+':0')  # input
                outputs = []
                for i in output_names:
                    outputs.append(sess.graph.get_tensor_by_name(i+':0'))
                out_pb = sess.run(outputs, feed_dict={input_x: img})

        print(f'out_pytorch {y}')
        print(f'out_onnx {out_onnx}')
        print(f'out_pb {out_pb}')
