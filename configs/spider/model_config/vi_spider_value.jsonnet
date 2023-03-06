local _data_path = 'data/vi-spider/';
local spider_base = import "spider_base_0512.libsonnet";

function(args, data_path=_data_path) spider_base(args, data_path=_data_path) {
    data: {
        local PREFIX = data_path + "raw/word-level/",

        train: {
            name: 'spider', 
            paths: [
              PREFIX + 'train.json'],
            tables_paths: [
              PREFIX + 'tables.json',
            ],
            db_path: PREFIX + 'database',
        },
        val: {
            name: 'spider', 
            paths: [PREFIX + 'dev.json'],
            tables_paths: [PREFIX + 'tables.json'],
            db_path: PREFIX + 'database',
        },
        test: {
            name: 'spider',
            paths: [PREFIX + 'test.json'],
            tables_paths: [PREFIX + 'tables.json'],
            db_path: PREFIX + 'database',
        }

    },
    model+:{
        encoder+: {
            name: 'spiderv3',
            question_encoder: ['shared-vi-emb', 'bilstm-native'],
            column_encoder: ['shared-vi-emb', 'bilstm-native-summarize'],
            table_encoder: ['shared-vi-emb', 'bilstm-native-summarize'],
            rat_config+: {
                name: "rat",
                num_heads: 8,
                num_layers: $.args.num_layers,
                enable_latent_relations: false,
            },
        },
        encoder_preproc+: {
            word_emb: null,
            vi_word_emb: {
                name: "vi_phow2v_embedder",
            },
            compute_sc_link: $.args.sc_link,
            compute_cv_link: $.args.cv_link,
            save_path: _data_path + 'spider-0409-phow2v-spacy-stanza,sc_link=%s,cv_link=%s' % [$.args.sc_link, $.args.cv_link],
            use_vi_vocab: true,
        },

        decoder+: {
            loss_type: $.args.loss_type,
        },

        decoder_preproc+: {
            min_freq: 32,
        },

    },

    optimizer: {
        name: $.args.opt,
        lr: 0.0,
    },

    lr_scheduler+: {
        name: $.args.lr_scheduler,
        start_lrs: $.args.lr,
        end_lr: $.args.end_lr,
        num_warmup_steps: 500,
    },
}
