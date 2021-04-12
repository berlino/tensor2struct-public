{
    local exp_id = 1,
    project: "ch_spider_bert",
    logdir: "log/ch_spider/multilingual_bert_%d" %exp_id,
    model_config: "configs/spider/model_config/ch_spider_bert.jsonnet",
    model_config_args: {
        # model
        num_layers: 6,
        sc_link: false,
        cv_link: false,
        loss_type: "softmax", # softmax, label_smooth

        # bert
        opt: "torchAdamw",   # bertAdamw, torchAdamw
        lr_scheduler: "bert_warmup_polynomial_group_v2", # bert_warmup_polynomial_group,bert_warmup_polynomial_grou_v2
        bert_token_type: true,
        bert_version: "bert-base-multilingual-uncased",
        bert_lr: 2e-5,   # for bert-large, lr should be smaller that bert-base

        # grammar
        include_literals: false,

        # training
        bs: 3,
        att: 0,
        lr: 5e-4,
        clip_grad: 3,
        num_batch_accumulated: 4,
        max_steps: 20000,
        save_threshold: 10000,
        use_bert_training: true,
        device: "cuda:0",

        # meta train
        meta_train_opt: "sgd",
        meta_train_lr: 1e-4,
        num_batch_per_train: 3,
        data_scheduler: "db_scheduler",
    },

    eval_section: "val",
    eval_type: "match", # exec does not make sense for ch_spider
    eval_method: "spider_beam_search_with_heuristic",
    eval_output: "ie_dir/spider_value",
    eval_beam_size: 3,
    eval_debug: false,
    eval_name: "bert_run_%d_%s_%s_%d" % [exp_id, self.eval_section, self.eval_method, self.eval_beam_size],

    local _end_step = $.model_config_args.max_steps / 1000,
    local _start_step = $.model_config_args.save_threshold / 1000,
    eval_steps: [ 1000 * x for x in std.range(_start_step, _end_step)],
}
