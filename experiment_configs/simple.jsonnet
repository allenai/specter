local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

local TRAIN_PATH = std.extVar("TRAIN_PATH");
// local TEST_PATH = std.extVar("TEST_PATH");
local DEV_PATH = std.extVar("DEV_PATH");
local BERT_VOCAB = std.extVar("BERT_VOCAB");
local BERT_MODEL = std.extVar("BERT_MODEL");
local BERT_WEIGHTS = std.extVar("BERT_WEIGHTS");
local LAZY = stringToBool(std.extVar("LAZY"));
local NUM_EPOCHS = std.parseInt(std.extVar("NUM_EPOCHS"));
local BATCH_SIZE = std.extVar("BATCH_SIZE");
local BERT_REQUIRES_GRAD=std.extVar("BERT_REQUIRES_GRAD");
local VOCAB_PATH=std.extVar("VOCAB_DIR");
local TRAIN_DATA_INSTANCES=std.parseInt(std.extVar("TRAINING_DATA_INSTANCES"));
local MAX_SEQ_LEN=std.parseInt(std.extVar("MAX_SEQ_LEN"));
local INCLUDE_VENUE=stringToBool(std.extVar('INCLUDE_VENUE'));
// local CUDA_DEVICE  = [std.parseInt(x) for x in std.split(std.extVar("CUDA_DEVICE"), ' ')];
local CUDA_DEVICE = std.extVar("CUDA_DEVICE");
{
    "dataset_reader": {
        "type": "specter_data_reader_pickled",
        "concat_title_abstract": true,
        "lazy": true,
        "max_sequence_length": MAX_SEQ_LEN,
        "token_indexers": {
            "bert": {
                "type": BERT_MODEL,
                "do_lowercase": true,
                "truncate_long_sequences": true,
                [if BERT_MODEL == 'bert-pretrained' && BERT_WEIGHTS != 'bert-pretrained' then "pretrained_model"]: BERT_VOCAB,
                [if BERT_MODEL == 'pretrained_transformer' then "model_name"]: BERT_WEIGHTS,
            }
        },
        "word_splitter": "bert-basic"
    },
    "iterator": {
        "type": "basic",
        "batch_size": BATCH_SIZE,
        "cache_instances": true
    },
    "model": {
        "type": "specter",
        "abstract_encoder": {
            "type": "boe",
            "embedding_dim": 768
        },
        "author_feedforward": {
            "activations": [
                "relu"
            ],
            "dropout": [
                0.2
            ],
            "hidden_dims": [
                10
            ],
            "input_dim": 12,
            "num_layers": 1
        },
        "author_id_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10,
                    "trainable": true
                }
            }
        },
        "author_position_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 2,
                    "trainable": true
                }
            }
        },
        "bert_finetune": true,
        "dropout": 0.25,
        "embedding_layer_norm": true,
        "feedforward": {
            "activations": [
                "relu"
            ],
            "dropout": [
                0
            ],
            "hidden_dims": [
                100
            ],
            "input_dim": 1586,
            "num_layers": 1
        },
        "ignore_authors": true,
        "layer_norm": true,
        "loss_distance": "l2-norm",
        "loss_margin": "1",
        "predict_mode": false,
        "include_venue": INCLUDE_VENUE,
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            [if BERT_MODEL != 'pretrained_transformer' then "embedder_to_indexer_map"]: {
                "bert": [
                    "bert",
                    "bert-offsets"
                ],
                "tokens": [
                    "tokens"
                ]
            },
            "token_embedders": {
                "bert": {
                    "type": BERT_MODEL,
                    [if BERT_MODEL == 'bert-pretrained' then "pretrained_model"]: BERT_WEIGHTS,
                    [if BERT_MODEL == 'pretrained_transformer' then "model_name"]: BERT_WEIGHTS,
                    "requires_grad": BERT_REQUIRES_GRAD
                }
            }
        },
        "title_encoder": {
            "type": "boe",
            "embedding_dim": 768
        },
        "venue_encoder": {
            "type": "boe",
            "embedding_dim": 50
        },
        "venue_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 50,
                    "trainable": true
                }
            }
        }
    },
    "train_data_path": TRAIN_PATH,
    "validation_data_path": DEV_PATH,
    // "test_data_path": TEST_PATH,
    "trainer": {
        "cuda_device": [
            CUDA_DEVICE
        ],
        "grad_clipping": 1,
        "gradient_accumulation_batch_size": 32,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "cut_frac": 0.1,
            "num_epochs": NUM_EPOCHS,
            "num_steps_per_epoch": TRAIN_DATA_INSTANCES / 32, 
        },
        "min_delta": "0",
        "model_save_interval": 10000,
        "num_epochs": NUM_EPOCHS,
        "optimizer": {
            "type": "bert_adam",
            "lr": "3e-5",
            "max_grad_norm": 1,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.bias",
                        "LayerNorm.weight",
                        "layer_norm.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "t_total": -1,
            "weight_decay": 0.01
        },
        "patience": 5,
        "should_log_learning_rate": true,
        "validation_metric": "-loss"
    },
    "vocabulary": {
        "directory_path": VOCAB_PATH
    }
}