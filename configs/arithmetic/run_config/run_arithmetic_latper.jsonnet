function(args={}) {
    project: "arithmetic", # wandb

    local _default_args = {
        exp_id: 0,
        att: 0,
        eval_section: "gen",  # test, gen
    },
    args: _default_args + args,

    logdir: "log/arithmetic/latper_%d" % $.args.exp_id,
    model_config: "configs/arithmetic/model_config/arithmetic_latper.jsonnet",

    model_config_args: {
        # data
        mode: "infix2postfix",

        # latper encoder and decoder
        syntax_encoder: ["emb", "bilstm"],  # one or two bilstm
        semantic_encoder: ["emb",],  # emb, or embd + two bilstm
        gumbel_temperature: null, # use null to deactive gumbel
        forward_relaxed: true,
        use_map_decode: true,  # whether to use discrete permutation 
        decoder_score_f: "linear",

        # model
        word_emb_size: 256,
        action_emb_size: 256,
        enc_recurrent_size: 256,
        dec_recurrent_size: 256,
        dropout: 0.1,

        # train
        bs: 16,
        att: $.args.att,
        lr: 1e-3,  # 1e-3
        end_lr: 1e-3,
        max_steps: 1000,
        save_threshold: 1000,
        device: "cuda:0",

    },

    eval_section: $.args.eval_section, 
    eval_method: "beam_search",
    eval_output: "ie_dir/arithmetic",
    eval_beam_size: 1,
    eval_debug: false,
    eval_name: "arithmetic_latper_%d_%s_%s_bs-%d_att-%d" % [$.args.exp_id, $.args.eval_section, self.eval_method, self.eval_beam_size, $.args.att],

    local _end_step = $.model_config_args.max_steps / 200,
    local _start_step = $.model_config_args.save_threshold / 200,
    eval_steps: [$.model_config_args.max_steps],
}
