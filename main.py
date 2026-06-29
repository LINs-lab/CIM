from condense.condense import main as synt_main
from data.utils.evaluator import eval_data

if __name__ == "__main__":
    from argument import args

    # synt_main(args)

    if args.dataset == 'imagenet-1k':
        epoch_eval = 300
    else:
        epoch_eval = 1000

    eval_data(
        save_dir=args.save_dir,
        model_ls=[args.model],
        tar_model_ls=[args.model],  #
        factor=args.factor,
        epochs=epoch_eval,  #
        batch_size=None,
        crop_method="factor",
        mix_type="vanilla",
        dsa_strategy="color_crop_cutout_flip_scale_rotate",
        store_log=True,
        eval_times=3,
        num_val=4,
        zca=False,
        logger_name='evaluation_log',  #
    )
