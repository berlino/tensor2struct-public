local _data_path = "data/overnight/";
local _raw_data_path = "data/overnight/semi_sup_split";
local _overnight_0315 = import "overnight_0315.jsonnet";
local _domains = ["blocks", "calendar", "publications", "housing", "recipes", "restaurants", "socialnetwork"];

function (args, data_path=_data_path) _overnight_0315(args, _data_path) {
    local data_split_path = "%s/%s" % [_raw_data_path, $.args.data_split],
    local target_domain = $.args.target_domain,
    local percent = $.args.percent,
    data: {
        train: {
            name: "overnight",
            paths: ["%s/%s/%s_train_%d.tsv" % [data_split_path, target_domain, target_domain, percent],],
        },
        unlabel_train: {
            name: "overnight",
            paths: ["%s/%s/%s_train_unlabelled_%d.tsv" % [data_split_path, target_domain, target_domain, percent],],
        },
        val: {
            name: "overnight",
            paths: ["%s/%s/%s_test.tsv" % [data_split_path, target_domain, target_domain],],
        },
    },

    model+: {
        name: "SemiSupEncDec",
        search_scheduler: {
            name: "online",
            sample_size: $.args.sample_size,
            use_cache: $.args.use_cache,
            use_gumbel: $.args.use_gumbel,
            ratio: $.args.ratio,
            top_p: $.args.top_p,
        },
        unsup_config:{
            enable_unsup_loss: $.args.enable_unsup_loss,
            unsup_loss_type: $.args.unsup_loss_type,
            alpha: $.args.alpha,
        },
        encoder_preproc+: {
           save_path: _data_path + 'overnight,semi_sup_%s,%s,sc_link=%s,cv_link=%s,percent=%d' % [target_domain, $.args.data_split, $.args.sc_link, $.args.cv_link, percent],
        },
    },

    train+: {
        eval_on_train: false,
        pretrain_threshold: $.args.pretrain_threshold,
        keep_every_n: 200,
    },

    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: $.args.pretrain_threshold,
        start_lr: $.args.lr,
        end_lr: 1e-5,
        decay_steps: $.train.max_steps - $.args.pretrain_threshold,
        power: 0.5,
    },
}
