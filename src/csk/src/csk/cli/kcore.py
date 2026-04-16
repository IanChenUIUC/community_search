import click

from .run_shell_baseline import shell_baseline_index, shell_baseline_search
from .run_shell_compressed import (
    shell_compressed_build,
    shell_compressed_index,
    shell_compressed_search,
)


@click.group()
def main():
    pass


main.add_command(shell_baseline_index)
main.add_command(shell_baseline_search)
main.add_command(shell_compressed_index)
main.add_command(shell_compressed_build)
main.add_command(shell_compressed_search)

if __name__ == "__main__":
    main()
