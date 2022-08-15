{
    local exp_id = 0,
    project: "spider_value",
    logdir: "log/spider/bert_value_%d" %exp_id,
    model_config: "configs/spider/model_config/vi_spider_bert_value.jsonnet",
    model_config_args: {
        # data 
        use_other_train: true,

        # model
        num_layers: 6,
        sc_link: false,
        cv_link: false,
        loss_type: "softmax", # softmax, label_smooth

        # bert
        opt: "torchAdamw",   # bertAdamw, torchAdamw
        lr_scheduler: "bert_warmup_polynomial_group_v2", # bert_warmup_polynomial_group,bert_warmup_polynomial_grou_v2
        bert_token_type: false,
        bert_version: "vinai/phobert-large",
        bert_lr: 2e-5, 

        # grammar
        include_literals: true,

        # training
        bs: 2,
        att: 0,
        lr: 5e-4,
        clip_grad: 0.3,
        num_batch_accumulated: 1,
        max_steps: 1000,
        save_threshold: 900,
        use_bert_training: true,
        device: "cuda:0",
    },

    eval_section: "val",
    eval_type: "all", # match, exec, all
    eval_method: "spider_beam_search_with_heuristic",
    eval_output: "ie_dir/spider_value",
    eval_beam_size: 3,
    eval_debug: false,
    eval_name: "bert_run_%d_%s_%s_%d_%d" % [exp_id, self.eval_section, self.eval_method, self.eval_beam_size, self.model_config_args.att],

    local _start_step = $.model_config_args.save_threshold / 1000,
    local _end_step = $.model_config_args.max_steps / 1000,
    eval_steps: [ 1000 * x for x in std.range(_start_step, _end_step)],
}
