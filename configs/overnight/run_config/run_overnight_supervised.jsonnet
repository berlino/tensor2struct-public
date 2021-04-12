{
    local exp_id = 0,
    project: "overnight_sup",
    target_domain: "basketball", # calendar, blocks, socialnetwork, publications, recipes, restaurants, housing, basketball
    logdir: "log/overnight-%s/sup_run_%s_%d" %[self.target_domain, self.target_domain, exp_id],
    model_config: "configs/overnight/model_config/overnight_0315.jsonnet",
    model_config_args: {
        # training
        bs: 12,
        att: 0,
        device: "cuda:0",
        max_steps: 5000,
        clip_grad: null,
        # save_threshold: self.max_steps / 2,
        save_threshold: 3000,

        # model 
        num_layers: 2,
        sc_link: true,
        cv_link: true,
        top_k_learnable: 50,
        target_domain: $.target_domain,
    },

    # eval_method: "beam_search",
    eval_method: "overnight_beam_search",
    eval_beam_size: 3,
    eval_output: "ie_dir/overnight",
    eval_debug: false,
    eval_name: "overnight-%s_%s_run_%d_%s_beam_%d" % [self.eval_section, self.target_domain, exp_id, self.eval_method, self.eval_beam_size],

    eval_steps: [ 5000 ],
    eval_section: "val",
}
