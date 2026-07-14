# About

The worry is that multiple jobs on the same machine will lead to noise in the runtime of the algorithms.
The main cause I am concerned about is if the page cache is pre-loaded beforehand (only for some jobs), from a previous job.

Co-tenancy (when multiple jobs are running on the same machine), can also cause problems as there is no virtualization on these machines.
There could also be heterogeneous CPU models across machines.
These are both plausible issues, with the first one being addressed by the `--exclusive` SLURM flag, and the latter by setting a [https://wiki.rc.usf.edu/index.php/SLURM_Using_Features_and_Constraints](contraint), `--constraint=cpu_xeon&sse4` for example.

This experiment is:

1. measuring the difference in the Icebug scripts for a cold vs warm run, via `vmtouch`
2. measuring whether this effect is seen from
  - running two jobs back-to-back on the same network
  - running two jobs at the same time on the same network

Conducted on `valhalla`, with `AMD EPYC 7702 64-Core Processor` CPU with 2 sockets and 256GB memory.
