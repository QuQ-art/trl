from train_minillm_dolly_gpt2 import build_parser, run


def main():
    run(build_parser(smoke_defaults=True).parse_args())


if __name__ == "__main__":
    main()
