from __future__ import annotations

from clize import run

from .commands import evaluate, export, infer, train, transfer


def main() -> None:
    run(
        {
            "train": train,
            "eval": evaluate,
            "infer": infer,
            "transfer": transfer,
            "export": export,
        }
    )


if __name__ == "__main__":
    main()
