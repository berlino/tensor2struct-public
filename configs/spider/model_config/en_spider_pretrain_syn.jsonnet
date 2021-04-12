local _data_path = 'data/spider/';
local spider_base = import "spider_base_0512.libsonnet";

function(args, data_path=_data_path) spider_base(args, data_path=_data_path) {
    data: {
        local PREFIX = data_path + "raw/",
        local ts = if $.args.use_other_train then
            ['spider', 'others']
        else
            ['spider'],

        train: {
            name: 'spider', 
            paths: [
              PREFIX + 'train_%s.json' % [s]
              for s in ts],
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

        syn_train: {
            name: 'spider',
            paths: [$.args.syn_data_path],
            tables_paths: [PREFIX + 'tables.json'],
            db_path: PREFIX + 'database',
        },
    },
    model+:{
        encoder: {
            name: 'spider-bert',
            bert_version: $.args.bert_version,
            bert_token_type: $.args.bert_token_type,
            linking_config: {
                name: "spider_string_matching",
            },
            rat_config: {
                name: "rat",
                num_heads: 8,
                num_layers: $.args.num_layers,
                enable_latent_relations: false,
            },
        },
        encoder_preproc: {
            context: {
                "name": "spider-bert",
                db_paths: [
                    _data_path + "raw/database"
                ],
            },
            bert_version: $.model.encoder.bert_version,
            compute_sc_link: $.args.sc_link,
            compute_cv_link: $.args.cv_link,
            save_path: _data_path + 'spider-0727-%s,other_train-%s,syn-%s,sc_link=%s,cv_link=%s' % [self.bert_version, $.args.use_other_train, $.args.syn_data_version, $.args.sc_link, $.args.cv_link],
        },

        decoder+: {
            enc_recurrent_size: 
                if ($.args.bert_version == "bert-large-uncased-whole-word-masking") 
                    || ($.args.bert_version == "bert-large-cased-whole-word-masking")
                    || ($.args.bert_version == "google/electra-large-discriminator")
                then 1024 else 768,
            loss_type: $.args.loss_type,
        },

        decoder_preproc+: {
            min_freq: 32,
            value_tokenizer: $.model.encoder_preproc.bert_version,
        },

    },

    train+: {
        use_kd_train: $.args.use_kd_train,
        lambda_mixture: $.args.lambda_mixture,
        check_syn_consistency: $.args.check_syn_consistency,
        data_scheduler: {
            name: $.args.data_scheduler,
            syn_alpha: $.args.syn_alpha,
        },
    },

    optimizer: {
        name: $.args.opt,
        lr: 0.0,
        bert_lr: 0.0,
    },

    lr_scheduler+: {
        name: $.args.lr_scheduler,
        start_lrs: [$.args.lr, $.args.bert_lr],
        end_lr: $.args.end_lr,
        num_warmup_steps: $.train.max_steps / 8,
    },

    meta_train:: null,
    meta_test:: null,
}
