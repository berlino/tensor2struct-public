local _data_path = 'data/arithmetic/';

function(args={}, data_path=_data_path) {

    local _default_args = {
        bs: 6,
        att: 0,
    },

    args: _default_args + args,
    
    model_name: 'bs=%(bs)d,lr=%(lr)s,att=%(att)d' % ({
        bs: $.args.bs,
        lr: $.args.lr,
        att: $.args.att,
    }),

    data: {
        local PREFIX = data_path + "raw/",
        train: {
            name: 'arithmetic',
            path: PREFIX + "train_10k.tsv",
            mode: $.args.mode,
        },
        val: {
            name: 'arithmetic',
            path: PREFIX + "val_5k.tsv",
            mode: $.args.mode,
        },
        test: {
            name: 'arithmetic',
            path: PREFIX + "test_5k.tsv",
            mode: $.args.mode,
        },
        gen: {
            name: 'arithmetic',
            path: PREFIX + "gen_5k.tsv",
            mode: $.args.mode,
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'latper_enc',
            dropout: $.args.dropout,
            word_emb_size: $.args.word_emb_size,
            recurrent_size: $.args.enc_recurrent_size,
            syntax_encoder: $.args.syntax_encoder,
            semantic_encoder: $.args.semantic_encoder,
            forward_relaxed: $.args.forward_relaxed,
            gumbel_temperature: $.args.gumbel_temperature,
            use_map_decode: $.args.use_map_decode,
        },
        encoder_preproc: {
            save_path: _data_path + "arithmetic-1222-latper",
        },
        decoder: {
            name: 'arithmetic_tagging_dec',
            score_f: $.args.decoder_score_f,
            enc_recurrent_size: $.model.encoder.recurrent_size,
        },
        decoder_preproc: {
            save_path: $.model.encoder_preproc.save_path,
        },
    },

    train: {
        device: $.args.device,
        batch_size: $.args.bs,
        eval_batch_size: $.args.bs,
        num_batch_accumulated: 1,
        clip_grad: null,

        model_seed: $.args.att,
        data_seed:  $.args.att,
        init_seed:  $.args.att,

        keep_every_n: 100,
        eval_every_n: 200,
        save_threshold: $.args.save_threshold,
        save_every_n: 100,
        report_every_n: 10,

        max_steps: $.args.max_steps,
        num_eval_items: 6,
    },

    meta_train:: null,

    optimizer: {
        name: 'adam',
        lr: $.args.lr,
    },

    // lr_scheduler: {
    //     name: 'noop',
    // }

    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: 0,
        start_lr: $.args.lr,
        end_lr: $.args.end_lr,
        decay_steps: $.train.max_steps,
        power: 0.5,
    }
}
