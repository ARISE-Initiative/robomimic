from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "bc_xfmr_film"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc_transformer_film.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    if args.env == "r2d2":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [{"path": p} for p in scan_datasets("~/Downloads/example_pen_in_cup", postfix="trajectory_im128.h5")],
            ],
            value_names=[
                "pen-in-cup",
            ],
        )
        generator.add_param(
            key="observation.modalities.obs.rgb",
            name="cams",
            group=130,
            values=[
                # ["camera/image/hand_camera_left_image"],
                ["camera/image/hand_camera_left_image", "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_2_left_image"],
            ],
            value_names=[
                # "wrist",
                "3cams",
            ]
        )
    elif args.env == "kitchen":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values_and_names=[
                # (get_ds_cfg("all"), "pnp-doors-human"),
                (get_ds_cfg("pnp_counter_to_cab"), "pnp-counter-to-cab-human"),
                (get_ds_cfg("pnp_cab_to_counter"), "pnp-cab-to-counter-human"),
                (get_ds_cfg("pnp_counter_to_sink"), "pnp-counter-to-sink-human"),
                (get_ds_cfg("pnp_sink_to_counter"), "pnp-sink-to-counter-human"),
                (get_ds_cfg("pnp_counter_to_microwave"), "pnp-counter-to-microwave-human"),
                (get_ds_cfg("pnp_microwave_to_counter"), "pnp-microwave-to-counter-human"),
                (get_ds_cfg(["open_door_single_hinge", "open_door_double_hinge"]), "open-door-human"),
                (get_ds_cfg(["close_door_single_hinge", "close_door_double_hinge"]), "close-door-human"),
            ]
        )

        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="backbone",
            group=1234,
            values=[
                "ResNet18ConvFiLM",
                # "ResNet50Conv",
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.feature_dimension",
            name="visdim",
            group=1234,
            values=[
                64,
                # 512,
            ],
        )
    elif args.env == "square":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [
                    {"path": "~/datasets/square/ph/square_ph_abs_tmp.hdf5"}, # replace with your own path
                ],
            ],
            value_names=[
                "square",
            ],
        )
    else:
        raise ValueError

    # change default settings: predict 10 steps into future
    generator.add_param(
        key="algo.transformer.pred_future_acs",
        name="predfuture",
        group=1,
        values=[
            True,
            # False,
        ],
        hidename=True,
    )
    generator.add_param(
        key="algo.transformer.supervise_all_steps",
        name="supallsteps",
        group=1,
        values=[
            True,
            # False,
        ],
        hidename=True,
    )
    generator.add_param(
        key="algo.transformer.causal",
        name="causal",
        group=1,
        values=[
            False,
            # True,
        ],
        hidename=True,
    )
    generator.add_param(
        key="train.seq_length",
        name="",
        group=-1,
        values=[10],
        hidename=True,
    )

    generator.add_param(
        key="algo.gmm.min_std",
        name="mindstd",
        group=271314,
        values=[
            0.03,
            #0.0001,
        ],
        hidename=True,
    )
    generator.add_param(
        key="train.max_grad_norm",
        name="maxgradnorm",
        group=18371,
        values=[
            # None,
            100.0,
        ],
        hidename=True,
    )
    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
