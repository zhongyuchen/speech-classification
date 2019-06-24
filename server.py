import flask
import os
from flask import Flask, request, send_from_directory, render_template
import configparser
from cnn import VGG
import torch
import json
import base64
import datetime
from mfcc import mfcc_one
import time


# def test(model):
def save_file(data):
    filename = str(datetime.datetime.now()).replace(' ', '_') + '.wav'
    with open(filename, 'wb') as f:
        ori_image_data = base64.b64decode(data, '-_')
        f.write(ori_image_data)
    return filename


if __name__ == "__main__":
    app = flask.Flask(__name__, static_folder='static')
    app.debug = True

    config = configparser.ConfigParser()
    config.read("config.ini")
    model_path = config['data']['model_path']

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = VGG(batch_norm=True, num_classes=20, init_weights=False).to(device)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    idx2word = ['数字', '语音', '语言', '识别', '中国',
                '总工', '北京', '背景', '上海', '商行',
                '复旦', '饭店', 'Speech', 'Speaker', 'Signal',
                'Process', 'Print', 'Open', 'Close', 'Project']

    @app.route('/', methods=['POST', 'GET'])
    def home():
        return render_template('recorder.html')

    @app.route('/infer', methods=['GET', 'POST'])
    def infer():
        if request.method == 'GET':
            data = request.args.get('data')
        else:
            data = request.form.get('data')

        # file = flask.request.files['file']
        # app.logger.debug(file.filename)
        # os.makedirs("upload", exist_ok=True)
        # save_to = "upload/{}".format(file.filename)
        # file.save(save_to)
        # return test(model, save_to)
        # filename = save_file(data)
        # data = "2019-6-23-19-10-5.wav"
        print(data, type(data))
        # time.sleep(10)
        # print('start')
        while True:
            try:
                feat = mfcc_one('', data, time_length=2.0)
                break
            except:
                pass
            # except exception as e:
            #     pass
        # print(feat)
        output = model(feat)
        pred = output.argmax(dim=1)

        # idx = idx.data.numpy()
        # print(idx)
        # idx = idx[0]
        # print(idx)
        # idx = np.argsort(-idx)
        # idx = np.argmax(idx)
        # word = idx2word[idx[0]] + ',' + idx2word[idx[1]] + ',' + idx2word[idx[2]]
        # word = idx2word[idx]
        word = idx2word[pred.data[0]]
        ret = {
            'word': word
        }
        return json.dumps(ret)

    # @app.route('/js/<path:path>')
    # def send_js(path):
    #     return send_from_directory('interface/js', path)

    app.run(host="127.0.0.1", port=8800)
