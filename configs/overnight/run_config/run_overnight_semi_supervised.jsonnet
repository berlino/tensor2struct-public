function(args={}) {
    project: "overnight_semisup_naacl",

    # args
    local _default_args = {
        exp_id: 0,
        att: 0,
        target_domain: "basketball",
    },
    args: _default_args + args,

    # model config
    target_domain: $.args.target_domain, # calendar, blocks, socialnetwork, publications, recipes, restaurants, housing, basketball
    logdir: "log/overnight-%s/semi_sup_%s_run_%d" %[self.target_domain, self.target_domain, $.args.exp_id],
    model_config: "configs/overnight/model_config/semisup_overnight_0415.jsonnet",

    model_config_args: {
        # data
        data_split: "split-0",
        percent: 30, # {10, 20, 30, 40, 50},
        target_domain: $.target_domain,

        # training
        bs: 12,
        att: $.args.att,
        lr: 1e-3,
        device: "cuda:0",
        max_steps: 6000, 
        clip_grad: null,
        save_threshold: 4000,
        pretrain_threshold: 3000,

        # model 
        num_layers: 2,
        sc_link: true,
        cv_link: true,

        # unsup loss 
        enable_unsup_loss: true,  # if false, doing supervised learning only
        unsup_loss_type: "sparse", #  {null, self-train, top-k, repulsion, gentle, sparse}
        alpha: 0.3,  # coefficient for unsup loss, {0.1, 0.3, 0.5}

        # search scheduler
        sample_size: 16,
        use_gumbel: false,  # if true, use stochastic beam search
        use_cache: false,  # if true, cache previously retrieved ones
        ratio: 0,  # use 0 to disable it
        top_p: 0,  # use 0 to disable it
    },

    # eval_method: "beam_search",
    eval_method: "overnight_beam_search",
    eval_beam_size: 3,
    eval_output: "ie_dir/overnight",
    eval_debug: true,
    eval_type: 'match',
    eval_name: "overnight-%s_%s_run_%d_%s_beam_%d" % [self.eval_section, self.target_domain, $.args.exp_id, self.eval_method, self.eval_beam_size],

    eval_steps: [$.model_config_args.max_steps],
    eval_section: "val",
}
