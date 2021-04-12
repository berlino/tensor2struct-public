{
    local exp_id = 0,
    project: "spider_syn_naacl",
    logdir: "log/spider/bert_value_syn_%d" %exp_id,
    model_config: "configs/spider/model_config/en_spider_pretrain_syn.jsonnet",
    model_config_args: {
        # data 
        use_other_train: true,
        local syn_ratio = 6,
        local syn_max_ac = 128,
        syn_data_version: "r%s-m%s" %[syn_ratio, syn_max_ac],
        syn_data_path: "experiments/sql2nl/data-spider-with-ssp-synthetic/new-examples-%s-%s.json" %[syn_ratio, syn_max_ac],

        # model
        num_layers: 6,
        sc_link: true,
        cv_link: true,
        loss_type: "label_smooth", # label_smooth, softmax

        # bert
        opt: "torchAdamw",   # bertAdamw, torchAdamw
        lr_scheduler: "bert_warmup_polynomial_group_v2", # bert_warmup_polynomial_group,bert_warmup_polynomial_grou_v2
        bert_token_type: true,
        bert_version: "google/electra-base-discriminator",
        bert_lr: 2e-5, 

        # grammar
        include_literals: true,

        # scheduler
        syn_alpha: 1,
        data_scheduler: "synthesized", # warmup_with_decay, warmup, random, supervised, synthesized

        # kd training
        use_kd_train: false, # enable kd_train with lambda = 1 to use the filtered syn data
        lambda_mixture: 1,  # use 1 to disable kd loss
        check_syn_consistency: false,

        # training
        bs: 10,
        att: 0,
        lr: 5e-4,
        clip_grad: 0.3,
        max_steps: 40000,
        save_threshold: 30000,
        num_batch_accumulated: 1,
        use_bert_training: true,
        device: "cuda:0",
    },

    # eval
    eval_section: "val",
    eval_type: "all", # match, exec, all
    eval_method: "spider_beam_search_with_heuristic",
    eval_output: "ie_dir/spider_value_syn",
    eval_beam_size: 3,
    eval_debug: false,
    eval_name: "bert_run_%d_%s_%s_%d" % [exp_id, self.eval_section, self.eval_method, self.eval_beam_size],

    local _end_step = $.model_config_args.max_steps / 1000,
    local _start_step = $.model_config_args.save_threshold / 1000,
    eval_steps: [ 1000 * x for x in std.range(_start_step, _end_step)],
}
