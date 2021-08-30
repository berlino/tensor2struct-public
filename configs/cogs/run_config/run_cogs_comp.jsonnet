function(args={}) {
    project: "COGS_acl", # wandb

    local _default_args = {
        exp_id: 2,
        att: 6,
        eval_section: "test",  # gen_samples, gen, test
    },
    args: _default_args + args,

    logdir: "log/cogs/batched_vanilla_%d" % $.args.exp_id,
    model_config: "configs/cogs/model_config/cogs_1229_batched_seq.jsonnet",

    model_config_args: {
        ## model config, composing modules for LSTM or Transformer
        # encoder: ["emb", "bilstm", "bilstm"],  # bilstm or lstm
        # encoder: ["emb", "bilstm", "cls_glue", "transformer"],
        encoder: ["emb", "cls_glue_p", "transformer", "transformer"],  # two bilstm, cls_glue_p + 2 transformer, bilstm + cls_glue + transformer
        # decoder: ["lstm"], 
        decoder: ["transformer"], # lstm, transformer

        word_emb_size: 256,  # 256, 512
        action_emb_size: 256,
        enc_recurrent_size: 256,
        dec_recurrent_size: 256,
        dropout: 0.1, 

        # train
        bs: 128,
        att: $.args.att,
        lr: 1e-3,
        end_lr: 1e-5,
        max_steps: 6000,
        save_threshold: 4000,
        device: "cuda:3",

        # kernel
        data_scheduler: "cogs_rand_tree_kernel_scheduler", # length_scheduler, cogs_edist_scheduler, cogs_kernel_scheduler, cogs_nl_kernel_scheduler, cog_rand_scheduler, cogs_rand_edist_scheduler, cogs_rand_kernel_scheduler
        kernel_name: "pt",
        norm_kernel: true,
        lamb: 1.0,
        mu: 1.0,

        # meta train
        temp: 4,
        topk: 1000,
        meta_train_lr: 1e-3,
        num_batch_per_train: 2,  # num - 1 for virtual test, not application for some schedulers
    },

    eval_section: $.args.eval_section,  # {val, test, gen, gen_cp_recursion, gen_pp_recursion, gen_toy}
    eval_method: "batched_greedy_search",
    eval_output: "ie_dir/cogs_batched_vanilla",
    eval_beam_size: 1,
    eval_debug: false,
    eval_name: "cogs_batched_vanilla_%d_%s_%s_bs-%d_att-%d" % [$.args.exp_id, $.args.eval_section, self.eval_method, self.eval_beam_size, $.args.att],

    local _end_step = $.model_config_args.max_steps / 200,
    local _start_step = $.model_config_args.save_threshold / 200,
    # eval_steps: [ $.model_config_args.max_steps ],
    eval_steps: [ 2000 ],
}
