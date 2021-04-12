local spider_base = import "spider_base_0512.libsonnet";
local _data_path = 'data/ch_spider/';

function(args, data_path=_data_path) spider_base(args, data_path=_data_path) {
    data: {
        local PREFIX = data_path + "raw/",
        train: {
            name: 'spider', 
            paths: [PREFIX + 'train.json'],
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
            save_path: _data_path + 'ch_spider-0920-%s,sc_link=%s,cv_link=%s' % [self.bert_version, $.args.sc_link, $.args.cv_link],
        },

        decoder+: {
            enc_recurrent_size: 
                # if std.length(std.findSubstr($.args.bert_version, "large")) > 0
                if std.startsWith($.args.bert_version, "bert-large")
                then 1024 else 768,
            loss_type: $.args.loss_type,
        },

        decoder_preproc+: {
            min_freq: 32,
            value_tokenizer: $.model.encoder_preproc.bert_version,
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
}
