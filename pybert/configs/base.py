from pathlib import Path

BASE_DIR = Path('pybert')
config = {
    'raw_data_path': BASE_DIR / 'dataset/cnews.txt',
    'test_path': BASE_DIR / 'dataset/test.txt',

    'data_dir': BASE_DIR / 'dataset',
    'log_dir': BASE_DIR / 'output/log',
    'writer_dir': BASE_DIR / "output/TSboard",
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'cache_dir': BASE_DIR / 'model/',
    'result': BASE_DIR / "output/result",

    'bert_vocab_path': BASE_DIR / 'pretrain/bert/base-chinese/vocab.txt',
    'bert_config_file': BASE_DIR / 'pretrain/bert/base-chinese/config.json',
    'bert_model_dir': BASE_DIR / 'pretrain/bert/base-chinese',

    'xlnet_vocab_path': BASE_DIR / 'pretrain/xlnet/base-cased/spiece.model',
    'xlnet_config_file': BASE_DIR / 'pretrain/xlnet/base-cased/config.json',
    'xlnet_model_dir': BASE_DIR / 'pretrain/xlnet/base-cased'
}
