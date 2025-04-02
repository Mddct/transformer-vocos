from absl import app, flags
from ml_collections import config_flags

from vocos.train_utils import VocosState

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file('config')


def main(_):
    config = FLAGS.config
    print(config)
    assert config.model_dir != ''
    assert config.tensorboard_dir != ''
    assert config.train_data != ''
    state = VocosState(config)
    state.train()


if __name__ == '__main__':
    app.run(main)
