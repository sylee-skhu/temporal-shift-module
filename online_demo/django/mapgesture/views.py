from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from PIL import Image
from typing import Tuple
import cv2
import threading
import numpy as np
import sys
import os
import time
import torch
import tvm
from mapgesture import demo

# Create your views here.
def home(request):
    context = {}

    return render(request, "home.html", context)

class GestureRunner():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()

        self.frame_w = 640
        self.frame_h = 480
        self.label_w = 640
        self.label_h = 48
        self.disp_w = self.frame_w
        self.disp_h = self.frame_h + self.label_h
        self.disp = np.zeros([self.disp_h, self.disp_w, 3]).astype('uint8') + 255

        self.SOFTMAX_THRES = 0
        self.HISTORY_LOGIT = True
        self.REFINE_OUTPUT = True
        self.transform = demo.get_transform()
        self.executor, self.ctx = demo.get_executor()
        self.buffer = demo.get_buffer(self.ctx)

        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        frame = self.disp
        _, frame = cv2.imencode('.jpg', frame) 
        return frame.tobytes()

    def update(self):
        t = None
        index = 0
        idx = 0
        history = [2]
        history_logit = []
        history_timing = []
        i_frame = -1
        while True:
            i_frame += 1
            (self.grabbed, self.frame) = self.video.read()
            if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
                t1 = time.time()
                img_tran = self.transform([Image.fromarray(self.frame).convert('RGB')])
                input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
                img_nd = tvm.nd.array(input_var.detach().numpy(), device=self.ctx)
                inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + self.buffer
                outputs = self.executor(inputs)
                feat, self.buffer = outputs[0], outputs[1:]
                assert isinstance(feat, tvm.nd.NDArray)
                
                if self.SOFTMAX_THRES > 0:
                    feat_np = feat.asnumpy().reshape(-1)
                    feat_np -= feat_np.max()
                    softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                    print(max(softmax))
                    if max(softmax) > self.SOFTMAX_THRES:
                        idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
                    else:
                        idx_ = idx
                else:
                    idx_ = np.argmax(feat.asnumpy(), axis=1)[0]

                if self.HISTORY_LOGIT:
                    history_logit.append(feat.asnumpy())
                    history_logit = history_logit[-12:]
                    avg_logit = sum(history_logit)
                    idx_ = np.argmax(avg_logit, axis=1)[0]

                idx, history = demo.process_output(idx_, history, self.REFINE_OUTPUT)

                t2 = time.time()
                print(f"{index} {demo.catigories[idx]}")


                current_time = t2 - t1

            frame = cv2.resize(self.frame, (self.frame_w, self.frame_h))
            frame = frame[:, ::-1]
            label = np.zeros([self.label_h, self.label_w, 3]).astype('uint8') + 255

            cv2.putText(label, 'Prediction: ' + demo.catigories[idx],
                        (0, int(self.label_h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
            cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                        (self.label_w - 170, int(self.label_h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

            self.disp = np.concatenate((frame, label), axis=0)

            if t is None:
                t = time.time()
            else:
                nt = time.time()
                index += 1
                t = nt


def gen(runner):
    while True:
        frame = runner.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def mapgesture(request):
    try:
        runner = GestureRunner()
        return StreamingHttpResponse(gen(runner), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("에러입니다...")
        pass
