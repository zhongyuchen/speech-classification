# speech-classification

CNN and VGG isolated word recognition with interactive website for testing, using `flask` as backend.

## Prerequisites

* Install the required packages by:
```angular2
pip install -r requirements.txt
```

## Dataset

The speech recognition neural network is trained on [DSPSpeech-20](https://github.com/czhongyu/DSPSpeech-20) dataset,
which is collected on this [website](https://czhongyu.github.io/audio-collector/). 
Check [here](https://github.com/czhongyu/audio-collector) for the implementation of the dataset collecting website.

## Speech Recognition Implementation

Check `report/report.pdf` for more details.

* train/dev ratio: 4-1
* feature: spectrogram and MFCC
* model: CNN and VGG

## How to train

* model choices in `['cnn', 'vgg']`, data choices in `['spectrogram', 'mfcc']`
```angular2
python run_cnn.py --model vgg --data mfcc
```

## Result: dev accuracy

| | CNN | VGG |
| ------ | ------ | ------ |
| spectroram| 74.5 | 83.75|
| MFCC | 82.25 | __94.4__ |

## Website Implementation

* backend: `flask`
* send data from js to python backend: `ajax`

## How to set up website for testing

```angular2
python server.py
```

## Author

Zhongyu Chen
