"""Merge a native deletion checkpoint to standard hard-masked HF weights.

Use the pinned installed verl merger's usual merge --backend fsdp arguments.
The native checkpoint stays resumable and untouched.
"""


def main():
    from kit.sdft_verl import check_pinned
    from deletion_tuning.deletion_mask import hard_state
    from verl.model_merger.base_model_merger import generate_config_from_args, parse_args
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger
    check_pinned()
    config = generate_config_from_args(parse_args())
    if config.backend != 'fsdp' or config.operation != 'merge' or config.hf_upload:
        raise ValueError('deletion export requires a local FSDP merge')

    class DeletionMerger(FSDPModelMerger):
        def save_hf_model_and_tokenizer(self, state):
            return super().save_hf_model_and_tokenizer(hard_state(state))

    merger = DeletionMerger(config)
    try:
        merger.merge_and_save()
    finally:
        merger.cleanup()


if __name__ == '__main__':
    main()
