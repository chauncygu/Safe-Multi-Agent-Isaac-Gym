


def process_ppo(args, env, cfg_train, logdir):
    from algorithms.ppo.ppo import PPO, ActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if not is_testing:
        is_testing = args.test
    if args.resume > 0:
        chkpt = args.resume

    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])

    """Set up the PPO system for training or inferencing."""
    ppo = PPO(vec_env=env,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    if is_testing:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        ppo.test("/{}/model_{}.pt".format(logdir, chkpt))
    elif chkpt > 0:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        ppo.load("{}/model_{}.pt".format(logdir, chkpt))

    return ppo



def process_sac(args, env, cfg_train, logdir):
    from algorithms.sac import SAC, MLPActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if not is_testing:
        is_testing = args.test
    if args.resume > 0:
        chkpt = args.resume

    """Set up the PPO system for training or inferencing."""
    sac = SAC(vec_env=env,
              actor_critic=MLPActorCritic,
              ac_kwargs = dict(hidden_sizes=[learn_cfg["hidden_nodes"]]* learn_cfg["hidden_layer"]),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              replay_size = learn_cfg["replay_size"] ,
              # clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              polyak = learn_cfg["polyak"],
              learning_rate = learn_cfg["learning_rate"],
              max_grad_norm = learn_cfg.get("max_grad_norm", 2.0),
              entropy_coef = learn_cfg["ent_coef"],
              use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False),
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    if is_testing:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        sac.test("/{}/model_{}.pt".format(logdir, chkpt))
    elif chkpt > 0:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        sac.load("{}/model_{}.pt".format(logdir, chkpt))

    return sac


def process_td3(args, env, cfg_train, logdir):
    from algorithms.td3 import TD3, MLPActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if not is_testing:
        is_testing = args.test
    if args.resume > 0:
        chkpt = args.resume

    """Set up the PPO system for training or inferencing."""
    td3 = TD3(vec_env=env,
              actor_critic=MLPActorCritic,
              ac_kwargs = dict(hidden_sizes=[learn_cfg["hidden_nodes"]]* learn_cfg["hidden_layer"]),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              replay_size = learn_cfg["replay_size"] ,
              # clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              polyak = learn_cfg["polyak"],
              learning_rate = learn_cfg["learning_rate"],
              max_grad_norm = learn_cfg.get("max_grad_norm", 2.0),
              policy_delay=learn_cfg["policy_delay"],#2,
              act_noise= learn_cfg["act_noise"], #0.1,
              target_noise=learn_cfg["target_noise"], #0.2,
              noise_clip= learn_cfg["noise_clip"], #0.5,
              use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False),
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    if is_testing:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        td3.test("/{}/model_{}.pt".format(logdir, chkpt))
    elif chkpt > 0:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        td3.load("{}/model_{}.pt".format(logdir, chkpt))

    return td3


def process_ddpg(args, env, cfg_train, logdir):
    from algorithms.ddpg import DDPG, MLPActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if not is_testing:
        is_testing = args.test
    if args.resume > 0:
        chkpt = args.resume

    """Set up the DDPG system for training or inferencing."""
    ddpg = DDPG(vec_env=env,
              actor_critic=MLPActorCritic,
              ac_kwargs = dict(hidden_sizes=[learn_cfg["hidden_nodes"]]* learn_cfg["hidden_layer"]),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              replay_size = learn_cfg["replay_size"] ,
              gamma=learn_cfg["gamma"],
              polyak = learn_cfg["polyak"],
              learning_rate = learn_cfg["learning_rate"],
              max_grad_norm = learn_cfg.get("max_grad_norm", 2.0),
              act_noise= learn_cfg["act_noise"], #0.1,
              target_noise=learn_cfg["target_noise"], #0.2,
              noise_clip= learn_cfg["noise_clip"], #0.5,
              use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False),
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    if is_testing:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        ddpg.test("/{}/model_{}.pt".format(logdir, chkpt))
    elif chkpt > 0:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        ddpg.load("{}/model_{}.pt".format(logdir, chkpt))

    return ddpg





def process_trpo(args, env, cfg_train, logdir):
    from algorithms.trpo import TRPO, ActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if not is_testing:
        is_testing = args.test
    if args.resume > 0:
        chkpt = args.resume

    """Set up the TRPO system for training or inferencing."""
    trpo = TRPO(vec_env=env,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              damping =learn_cfg["damping"],
              cg_nsteps =learn_cfg["cg_nsteps"],
              max_kl= learn_cfg["max_kl"],
              max_num_backtrack=learn_cfg["max_num_backtrack"],
              accept_ratio=learn_cfg["accept_ratio"],
              step_fraction=learn_cfg["step_fraction"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    if is_testing:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        trpo.test("/{}/model_{}.pt".format(logdir, chkpt))
    elif chkpt > 0:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        trpo.load("{}/model_{}.pt".format(logdir, chkpt))

    return trpo