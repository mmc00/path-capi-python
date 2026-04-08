from __future__ import annotations

from path_capi_python import PATHLoader


def main() -> None:
    loader = PATHLoader.from_environment()
    runtime = loader.load()

    print("PATH version:", loader.version(runtime))
    print("License check (10 vars, 10 nnz):", loader.check_license(runtime, 10, 10))


if __name__ == "__main__":
    main()
