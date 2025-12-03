import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from mnist_classifier.models.fc import ImageClassifier
from mnist_classifier.trainer import Trainer

from mnist_classifier.utils import load_mnist
from mnist_classifier.utils import split_data
from mnist_classifier.utils import get_model


def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required=True)  # 모델 weight 파일이 저장될 경로
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)  # 학습할 GPU 번호

    p.add_argument('--train_ratio', type=float, default=0.8)  # 학습 데이터 내에서 검증 데이터가 차지할 비율

    p.add_argument('--batch_size', type=int, default=256)  # 미니배치 크기
    p.add_argument('--n_epochs', type=int, default=20)  # epoch 수

    p.add_argument('--model', default="fc", choices=['fc', 'cnn'])  # 어떤 모델로 분류기를 학습할지 결정

    p.add_argument('--n_layers', type=int, default=5)  # 모델 레이어 개수
    p.add_argument('--use_dropout', action='store_true')  # 드롭아웃 사용 여부
    p.add_argument('--dropout_p', type=float, default=0.3)  # 드롭아웃 확률

    p.add_argument('--verbose', type=int, default=1)  # 학습 시 로그 출력의 정도

    config = p.parse_args()

    return config

def main(config):
    if config.gpu_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % config.gpu_id)
    
    # load data and split into train/valid dataset
    x, y = load_mnist(is_train=True, flatten=(config.model == 'fc'))  # config.model이 fc일 때만 벡터로 변환
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)
    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    # input, output size for the given dataset
    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    model = get_model(input_size, output_size, config, device).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(criterion)
    
    # initialize trainier
    trainer = Trainer(model, optimizer, criterion)

    # start training with the given dataset and configuration
    trainer.train(train_data=(x[0], y[0]), valid_data=(x[1], y[1]), config=config)

    # save best model weights
    torch.save({'model': trainer.model.state_dict(), 'opt': optimizer.state_dict(), 'config': config}, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)