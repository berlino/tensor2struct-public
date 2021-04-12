# args of the function will have a default value
# so that we could balance flexibility and inherintance
local _data_path = 'data/spider/';

function(args={}, data_path=_data_path) {
    
    # default args
    local _default_args = {
        # model
        num_layers: 4,
        sc_link: false,
        cv_link: false,

        # training
        bs: 6,
        att: 0,
        use_bert_training: false,
        clip_grad: null,
        lr: 0.000743552663260837,
        end_lr: 0,
        save_threshold: 0,
        max_steps: 20000,
        device: "cuda:0",
        num_batch_accumulated: 1,

        # matching config
        use_latent_relations: false,
        use_sinkhorn: true,
        num_latent_relations: 3,
        discrete_relation: false,
        norm_relation: false,
        symmetric_relation: true,
        combine_latent_relations: false,
        use_align_loss: false,

        # grammar
        include_literals: false,

        # pretrain
        pretrain_path: null,
        checkpoint_step: null,

        # maml setting
        meta_opt: "sgd",
        meta_lr: 1e-4,
        num_batch_per_train: 3,
        first_order: false,
        use_mask: false,
        mask_type: "l0",
        slow_parameters: null,
        data_scheduler: "mixed_db_scheduler",
    },

    # merge args, to support this, you have to use $.args in your inherited function
    args: _default_args + args,

    # format model name
    local lr_s = '%0.1e' % $.args.lr,
    local end_lr_s = '0e0',
    model_name: 'bs=%(bs)d,lr=%(lr)s,end_lr=%(end_lr)s,att=%(att)d' % ({
        bs: $.args.bs,
        lr: lr_s,
        end_lr: end_lr_s,
        att: $.args.att,
    }),

    data: {
        local PREFIX = data_path + "raw/",
        train: {
            name: 'spider', 
            paths: [
              PREFIX + 'train_%s.json' % [s]
              for s in ['spider', 'others']],
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

    model: {
        name: 'EncDecV2',
        encoder: {
            name: 'spiderv3',
            word_emb_size: 300,
            recurrent_size: 256,
            question_encoder: ['shared-en-emb', 'bilstm-native'],
            column_encoder: ['shared-en-emb', 'bilstm-native-summarize'],
            table_encoder: ['shared-en-emb', 'bilstm-native-summarize'],

            linking_config: 
                if $.args.use_latent_relations && $.args.use_sinkhorn then {
                    name: "sinkhorn_matching",
                    num_latent_relations: $.args.num_latent_relations,
                    combine_latent_relations: $.args.combine_latent_relations,
                } else if $.args.use_latent_relations then {
                    name: "bilinear_matching",
                    num_latent_relations: $.args.num_latent_relations,
                    discrete_relation: $.args.discrete_relation,
                    norm_relation: $.args.norm_relation,
                    symmetric_relation: $.args.symmetric_relation,
                    combine_latent_relations: $.args.combine_latent_relations,
                } else {
                    name: "spider_string_matching",
                },
            rat_config: {
                name: 'rat',
                num_heads: 8,
                num_layers: $.args.num_layers,
                enable_latent_relations: $.args.use_latent_relations,
                num_latent_relations: $.args.num_latent_relations,
                combine_latent_relations: $.args.combine_latent_relations
            },
            top_k_learnable: 50,
        },
        encoder_preproc: {
            word_emb: {
                name: 'glove_spacy',
                kind: '42B',
                lemmatize: true,
            },
            min_freq: 4,
            max_count: 5000,
            context: {
                name: "spider",
                db_path: _data_path + "raw/database",
            },
            compute_sc_link: $.args.sc_link,
            compute_cv_link: $.args.cv_link,
            count_tokens_in_word_emb_for_vocab: true,
            save_path: _data_path + 'spider-0409-glove-spacy-stanza,sc_link=%s,cv_link=%s' % [$.args.sc_link, $.args.cv_link],
        },
        decoder_preproc: {
            grammar: {
                name: "spiderv2",
                include_literals: $.args.include_literals,
                end_with_from: true,
                infer_from_conditions: true,
            }, 
            use_seq_elem_rules: true,
            min_freq: 3,
            max_count: 5000,
            save_path: $.model.encoder_preproc.save_path,
        },   
        decoder: {
            name: 'NL2CodeV2',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            recurrent_size : 512,
            loss_type: "softmax",
            use_align_mat: true,
            use_align_loss: $.args.use_align_loss,
            enumerate_order: false,
            enc_recurrent_size: $.model.encoder.recurrent_size,
        },
    },

    train: {
        use_bert_training: $.args.use_bert_training,
        device: $.args.device,
        batch_size: $.args.bs,
        eval_batch_size: $.args.bs,
        num_batch_accumulated: $.args.num_batch_accumulated,
        clip_grad: $.args.clip_grad,

        model_seed: $.args.att,
        data_seed:  $.args.att,
        init_seed:  $.args.att,

        keep_every_n: 1000,
        eval_every_n: 100,
        save_threshold: $.args.save_threshold,
        save_every_n: 100,
        report_every_n: 10,

        max_steps: $.args.max_steps,
        num_eval_items: 50,
    },

    meta_train: $.train + {
        inner_opt: {
            name: $.args.meta_opt,
            lr: $.args.meta_lr,
        },
        data_scheduler: {
            name: $.args.data_scheduler,
            batch_size: $.train.batch_size,
            num_batch_per_train: $.args.num_batch_per_train,
            use_similarity: true
        },
        first_order: $.args.first_order,
        // use_param_mask: $.args.use_mask,
        // mask_type: $.args.mask_type,
        // slow_parameters: $.args.slow_parameters,
    },

    meta_test: $.meta_train + {
        data_scheduler: {
            name: "random_scheduler_meta_test",
            batch_size: $.train.batch_size,
        },
        keep_every_n: 100,
    },

    pretrain: {
        pretrain_path: $.args.pretrain_path,
        checkpoint_step: $.args.checkpoint_step,
    },

    optimizer: {
        name: 'adam',
        lr: 0,
    },

    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: $.train.max_steps / 20,
        start_lr: $.args.lr,
        end_lr: $.args.end_lr,
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
    }
}
