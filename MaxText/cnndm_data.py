"""Language Model configurations on the T5/C4 dataset."""

import functools
import math
from typing import Dict, List, Optional, Sequence

from absl import logging
import jax
from jax import numpy as jnp

import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics

from jax.sharding import PartitionSpec as P
import tensorflow as tf

GPT_SPM_PATH = (
    'gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model'
)
GPT_EOS_ID = 1
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
PASS_THROUGH_VOCABULARY = t5.data.PassThroughVocabulary(size=50257)

C4_GPT_TRAIN_FEATURES_LM = {
    'inputs': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=False),
    'targets': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=False)
}

C4_GPT_EVAL_FEATURES_LM = {
    'inputs': t5.data.Feature(
        vocabulary=PASS_THROUGH_VOCABULARY, add_eos=False
    ),
    'targets': t5.data.Feature(
        vocabulary=PASS_THROUGH_VOCABULARY, add_eos=False
    )
}
TRAIN_DATADIR = 'gs://test-example-123/cnndm/'
EVAL_DATADIR = 'gs://test-example-123/cnndm/'
CNN_DATADIR = 'gs://test-example-123/datasets'

class TaskRegistry(t5.data.TaskRegistry):
  """Task registry with extra tracking."""

  TASK_NAMES = []

  @classmethod
  def add_versioned_tfds_task(cls,
                              name: str,
                              *,
                              versions: List[str],
                              pinned_version: Optional[str] = None,
                              tfds_name: str,
                              tfds_data_dir: Optional[str] = None,
                              **kwargs) -> List[seqio.Task]:
    tasks = []
    for version in versions:
      tasks.append(
          cls.add(
              f'{name}_{version}',
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    if pinned_version is not None:
      tasks.append(
          cls.add(
              name,
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{pinned_version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    return tasks

# C4 corpus for language model pretraining
TaskRegistry.add_versioned_tfds_task(
    name="cnn_dailymail_v001",
    versions=['3.4.0'],
    pinned_version='3.4.0',
    tfds_name='cnn_dailymail',
    tfds_data_dir=CNN_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.summarize,
            article_key='article',
            summary_key='highlights',
        ),
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=C4_GPT_TRAIN_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=10000,
)

TaskRegistry.add_versioned_tfds_task(
    name="cnn_dailymail_v001_eval",
    versions=['3.4.0'],
    pinned_version='3.4.0',
    tfds_name='cnn_dailymail',
    tfds_data_dir=CNN_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.summarize,
            article_key='article',
            summary_key='highlights',
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=C4_GPT_TRAIN_FEATURES_LM,
    metric_fns=[t5_metrics.bleu, t5_metrics.rouge],
    shuffle_buffer_size=10000,
)

def _dataset_common(is_training:bool, percore_batch_size: int, TRAINING_SEED=8731, MAX_SEQ_LEN=2048, OUTPUTS_LENGTH=256, TRAINING_NUM_BATCHES_TO_SKIP=False, shard_info=None) -> tf.data.Dataset:

    num_local_devices = jax.local_device_count()
    global_batch_size = int(
        percore_batch_size * num_local_devices * jax.process_count() + 1e-6
    )
    if percore_batch_size >= 1:
      assert global_batch_size % num_local_devices == 0
      batch_size_per_process = int(
          math.ceil(percore_batch_size) * num_local_devices + 1e-6
      )
      num_infeed_hosts = global_batch_size // batch_size_per_process
    else:
      if jax.process_count() > 1:
        assert global_batch_size % num_local_devices == 0
        batch_size_per_process = num_local_devices
        num_infeed_hosts = global_batch_size // batch_size_per_process
      else:
        batch_size_per_process = int(
            percore_batch_size * num_local_devices + 1e-6
        )
        num_infeed_hosts = 1
    seed = None
    if is_training:
      seed = TRAINING_SEED
      # TODO(sgpyc): enable sync of seeds across hosts, currently the
      # following failed because of "sync_global_devices name mismatch"
      # seed = jnp.int32(multihost_utils.broadcast_one_to_all(seed))
      logging.info('Train input seed: %s',
                   'None' if seed is None else seed)
      
    dataset = seqio.get_dataset(
      mixture_or_task_name='cnn_dailymail_v001_3.4.0' if is_training else 'cnn_dailymail_v001_eval_3.4.0',
      task_feature_lengths={
      'inputs': MAX_SEQ_LEN,
      'targets': OUTPUTS_LENGTH},         
      dataset_split='train' if is_training else 'validation',
      use_cached=False,
      num_epochs=None,
      feature_converter=seqio.seqio.DecoderFeatureConverter(
          pack=False
      ),
      shuffle=True if is_training else False,
      batch_size=batch_size_per_process,
      shard_info = shard_info,
      seed = seed,
    )
    return dataset

def get_cnndm_datasets(
  percore_batch_size
):
  train_ds = _dataset_common(is_training=True, percore_batch_size=percore_batch_size, TRAINING_SEED=1234)
  # shard the dataset as soon as it is loaded
  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index())

  eval_ds = _dataset_common(is_training=False, percore_batch_size=percore_batch_size)
  eval_ds = eval_ds.shard(num_shards = jax.process_count(), index = jax.process_index())

  return train_ds, eval_ds

def make_cnndm_train_iterator_and_tokenizer(config):
  """ Make train iterator and tokenizer for cnndm dataset"""
  train_ds, eval_ds = get_cnndm_datasets(
    percore_batch_size=1
  )

  def filter_keys(record):
    return {'inputs': record['decoder_input_tokens'], 'targets': record['decoder_target_tokens'],
    'inputs_segmentation':record['decoder_causal_attention'], 
    }
    
  train_ds = train_ds.map(filter_keys,num_parallel_calls=tf.data.AUTOTUNE)
  eval_ds = eval_ds.map(filter_keys,num_parallel_calls=tf.data.AUTOTUNE)


  
  return train_ds, eval_ds
    