# BERT Chinese text classification by PyTorch

This repo contains a PyTorch implementation of a pretrained BERT model  for text classification.

## Structure of the code

At the root of the project, you will see:

```text
├── pybert
|  └── callback
|  |  └── lrscheduler.py　　
|  |  └── trainingmonitor.py　
|  |  └── ...
|  └── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  └── dataset　　　
|  └── io　　　　
|  |  └── dataset.py　　
|  |  └── data_transformer.py　　
|  └── model
|  |  └── nn　
|  |  └── pretrain　
|  └── output #save the ouput of model
|  └── preprocessing #text preprocessing 
|  └── train #used for training a model
|  |  └── trainer.py 
|  |  └── ...
|  └── utils # a set of utility functions
├── convert_tf_checkpoint_to_pytorch.py
├── train_bert.py
```
## Dependencies

- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.0
- matplotlib
- tensorboardX
- Tensorflow (to be able to run TensorboardX)

## How to use the code

you need download pretrained chinese bert model

1. Download the Chinese BERT pretrained podel from [google search](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) and place it into the `/pybert/model/pretrain` directory.
2. `pip install pytorch-pretrained-bert` from [github](https://github.com/huggingface/pytorch-pretrained-BERT).
3. run `python convert_tf_checkpoint_to_pytorch.py` to transfer the pretrained model(tensorflow version)  into pytorch form .
4. prepare Chinese raw data(example,news data), you can modify the `io.data_transformer.py` to adapt your data.
5. Modify configuration information in `pybert/config/basic_config.py`(the path of data,...).
6. run `python train_bert`.

## Fine-tuning result

### training 

Epoch: 3 - loss: 0.0222 acc: 0.9939 - f1: 0.9911 val_loss: 0.0785 - val_acc: 0.9799 - val_f1: 0.9800

### classify_report

|    label    | precision | recall | f1-score | support |
| :---------: | :-------: | :----: | :------: | :-----: |
|     财经      |   0.97    |  0.96  |   0.96   |  1500   |
|     体育      |   1.00    |  1.00  |   1.00   |  1500   |
|     娱乐      |   0.99    |  0.99  |   0.99   |  1500   |
|     家居      |   0.99    |  0.99  |   0.99   |  1500   |
|     房产      |   0.96    |  0.97  |   0.96   |  1500   |
|     教育      |   0.98    |  0.97  |   0.97   |  1500   |
|     时尚      |   0.99    |  0.98  |   0.99   |  1500   |
|     时政      |   0.97    |  0.98  |   0.98   |  1500   |
|     游戏      |   1.00    |  0.99  |   0.99   |  1500   |
|     科技      |   0.96    |  0.97  |   0.97   |  1500   |
| avg / total |   0.98    |  0.98  |   0.98   |  15000  |

### training figure

![]( https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/20190214204557.PNG)

## Tips

- When converting the tensorflow checkpoint into the pytorch, it's expected to choice the "bert_model.ckpt", instead of "bert_model.ckpt.index", as the input file. Otherwise, you will see that the model can learn nothing and give almost same random outputs for any inputs. This means, in fact, you have not loaded the true ckpt for your model
- When using multiple GPUs, the non-tensor calculations, such as accuracy and f1_score, are not supported by DataParallel instance
- As recommanded by Jocob in his paper <url>https://arxiv.org/pdf/1810.04805.pdf<url/>, in fine-tuning tasks, the hyperparameters are expected to set as following: **Batch_size**: 16 or 32, **learning_rate**: 5e-5 or 2e-5 or 3e-5, **num_train_epoch**: 3 or 4
- The pretrained model has a limit for the sentence of input that its length should is not larger than 512, the max position embedding dim. The data flows into the model as: Raw_data -> WordPieces -> Model. Note that the length of wordPieces is generally larger than that of raw_data, so a safe max length of raw_data is at ~128 - 256 
- Upon testing, we found that fine-tuning all layers could get much better results than those of only fine-tuning the last classfier layer. The latter is actually a feature-based way 
