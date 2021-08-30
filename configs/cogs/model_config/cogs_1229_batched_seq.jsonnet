local _data_path = 'data/cogs/';

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
        local PREFIX = data_path + "raw/cogs/",
        train: {
            name: 'cogs',
            path: PREFIX + "train.tsv",
        },
        val: {
            name: 'cogs',
            path: PREFIX + "dev.tsv",
        },
        test: {
            name: 'cogs',
            path: PREFIX + "test.tsv",
        },
        gen_samples: {
            name: 'cogs',
            path: PREFIX + "gen_samples.tsv",
        },
        gen: {
            name: 'cogs',
            path: PREFIX + "gen.tsv",
        },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'cogs_enc',
            encoder: $.args.encoder,
            dropout: $.args.dropout,
            word_emb_size: $.args.word_emb_size,
            recurrent_size: $.args.enc_recurrent_size,
        },
        encoder_preproc: {
            save_path: _data_path + "cogs-1229-batched-vanilla",
        },
        decoder: {
            name: 'cogs_%s_dec' % $.args.decoder,
            enc_recurrent_size: $.model.encoder.recurrent_size,
            recurrent_size: $.args.dec_recurrent_size,
            action_emb_size: $.args.action_emb_size,
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

        keep_every_n: 200,
        eval_every_n: 200,
        save_threshold: $.args.save_threshold,
        save_every_n: 200,
        report_every_n: 200,

        max_steps: $.args.max_steps,
        num_eval_items: 50,
    },

    meta_train: $.train + {
        inner_opt: {
            name: "sgd",
            lr: $.args.meta_train_lr,
        },
        data_scheduler: {
            name: $.args.data_scheduler,
            batch_size: $.train.batch_size,
            num_batch_per_train: $.args.num_batch_per_train,
            topk: $.args.topk,
            temp: $.args.temp,
            kernel_name: $.args.kernel_name,
            norm_kernel: $.args.norm_kernel,
            lamb : $.args.lamb,
            mu: $.args.mu,
        },
    },

    optimizer: {
        name: 'adam',
        lr: 0,
    },

    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: 0,
        start_lr: $.args.lr,
        end_lr: $.args.end_lr,
        decay_steps: $.train.max_steps,
        power: 0.5,
    }

}