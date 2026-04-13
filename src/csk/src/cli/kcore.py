import click

from .run_shell_baseline import shell_baseline_index, shell_baseline_search
from .run_shell_compressed import (
    shell_compressed_build,
    shell_compressed_index,
    shell_compressed_search,
)


@click.group()
def kcore():
    pass


kcore.add_command(shell_baseline_index)
kcore.add_command(shell_baseline_search)
kcore.add_command(shell_compressed_index)
kcore.add_command(shell_compressed_build)
kcore.add_command(shell_compressed_search)

if __name__ == "__main__":
    kcore()
