local _data_path = "data/overnight/";
local _raw_data_path = "data/overnight/jonathan-full/";
# local _domains = ["blocks", "calendar", "publications", "basketball", "housing", "recipes", "restaurants", "socialnetwork"];
local _domains = ["blocks", "calendar", "publications", "housing", "recipes", "restaurants", "socialnetwork"];

function(args, data_path = _data_path) {
    # default args
    local _default_args = {
        # model
        num_layers: 4,
        sc_link: false,
        cv_link: false,
        top_k_learnable: 50,

        # training
        bs: 6,
        att: 0,
        clip_grad: null,
        lr: 0.000743552663260837,
        end_lr: 0,
        save_threshold: 0,
        max_steps: 20000,
        device: "cuda:0",
        num_batch_accumulated: 1,

        # pretrain
        pretrain_path: null,
        checkpoint_step: null,
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

    local target_domain = $.args.target_domain,
    data: {
        local _cross_domains = std.filter(function(x) if x == target_domain then false else true, _domains),
        train: {
            name: "overnight",
            paths: [_raw_data_path + target_domain + "_train.tsv"],
        },
        val: {
            name: "overnight",
            paths: [_raw_data_path + target_domain + "_test.tsv"],
        },
    },


    model: {
        name: 'EncDecV2',
        encoder: {
            name: 'overnight',
            word_emb_size: 300,
            recurrent_size: 256,
            # batch_encs_update: false,
            question_encoder: ['emb', 'bilstm-native'],
            column_encoder: ['emb', 'bilstm-native-summarize'],
            value_encoder: ['emb', 'bilstm-native-summarize'],
            linking_config: {
                name: "overnight_string_matching",
            },
            rat_config: {
                name: 'rat',
                num_heads: 8,
                num_layers: $.args.num_layers,
                enable_latent_relations: false,
            },
            top_k_learnable: $.args.top_k_learnable,
        },
        encoder_preproc: {
            word_emb: {
                name: 'glove_spacy',
                kind: '42B',
                lemmatize: true,
            },
            grammar: $.model.decoder_preproc.grammar,
            context: {
                name: "overnight",
            },
            min_freq: 3,
            max_count: 8000,
            sc_link: $.args.sc_link,
            cv_link: $.args.cv_link,
            save_path: data_path + "overnight,target_domain-%s,0401,emb=glove-42B,min_freq=%s,max_count=%s,sc_link=%s,cv_link=%s" %[$.args.target_domain, self.min_freq, self.max_count, $.args.sc_link, $.args.cv_link],
        },
        decoder: {
            name: "overnight_decoder",
            rule_emb_size: 128,
            enc_recurrent_size: $.model.encoder.recurrent_size,
        },
        decoder_preproc: {
            grammar: {
                name: "overnight",
            },
            save_path: $.model.encoder_preproc.save_path,
        },
    },

    train: {
        device: $.args.device,
        batch_size: $.args.bs,
        eval_batch_size: $.args.bs,
        num_batch_accumulated: $.args.num_batch_accumulated,
        clip_grad: $.args.clip_grad,

        model_seed: $.args.att,
        data_seed:  $.args.att,
        init_seed:  $.args.att,

        keep_every_n: 200,
        eval_every_n: 100,
        save_threshold: $.args.save_threshold,
        save_every_n: 100,
        report_every_n: 10,

        max_steps: $.args.max_steps,
        num_eval_items: 50,
    },

    optimizer: {
        name: 'adam',
        lr: $.args.lr,
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
