library(dplyr)
library(tidyverse)
library(ggh4x)
library(xtable)

## ================== icebug versus python

data <- read.csv("py_v_ib.csv")
data |> head()
lapply(data |> select(-c("wall_s")), unique)

data |>
  group_by(network, algorithm, query_size, framework) |>
  summarize(n = n())

data |>
  mutate(
    algorithm = recode(algorithm,
      "shell_offline" = "shell (offline)",
      "shell_online" = "shell (online)",
      "steiner" = "steiner (online)"
    )
  ) |>
  ggplot(aes(x = network, y = wall_s, fill = framework)) +
  facet_grid(
    rows = vars(algorithm),
    scales = "free_y"
  ) +
  scale_x_discrete(name = NULL) +
  scale_y_continuous(name = "Runtime (s)") +
  geom_col(
    position = position_dodge2(width = 0.9, preserve = "single"),
    data = ~ filter(.x, algorithm == "shell (offline)")
  ) +
  geom_boxplot(
    data = ~ filter(.x, algorithm != "shell (offline)")
  ) +
  theme_bw() +
  theme(
    strip.text = element_text(size = 8),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.text = element_text(size = 12),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.margin = margin(t = -3, unit = "mm")
  )

ggsave("py_v_ib.pdf", width = 122, height = 80, units = "mm")

## ================== strong scaling of icebug

data <- rbind(
  read.csv("abm14_curve.csv") |> mutate(network = "abm14"),
  read.csv("cen_curve.csv") |> mutate(network = "cen")
)
data |> head()
lapply(data |> select(threads, network), unique)

data |>
  mutate(total = coredecomp + shellstruct, mem = mem / 1e6) |>
  pivot_longer(!c("network", "threads")) |>
  mutate(row_group = if_else(name == "mem", "memory (GB)", "time (s)")) |>
  ggplot(aes(x = threads, y = value)) +
  geom_line(aes(color = name), data = ~ filter(.x, row_group == "time (s)")) +
  geom_point(aes(color = name), data = ~ filter(.x, row_group == "time (s)")) +
  geom_col(aes(width = 1), data = ~ filter(.x, row_group == "memory (GB)")) +
  facet_grid(
    rows = vars(row_group),
    cols = vars(network),
    scales = "free_y",
  ) +
  scale_x_continuous(name = "number of cpus", breaks = c(1, 2, 4, 8, 16, 32, 64)) +
  facetted_pos_scales(
    y = list(
      row_group == "time (s)" ~ scale_y_continuous(name = NULL, limits = c(0, 100)),
      row_group == "memory (GB)" ~ scale_y_continuous(name = NULL)
    )
  ) +
  theme_bw() +
  theme(
    strip.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.text = element_text(size = 12),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.margin = margin(t = -3, unit = "mm")
  )
ggsave("shellstruct_strong_scaling.pdf", width = 122, height = 80, units = "mm")

## ================== community search testing

cs_failed <- bind_rows(
  expand_grid(
    network = c("friendster", "twitter_social", "wikipedia_link"),
    method = "shellstruct",
    query_type = c("multi", "single"),
    batch_type = c("batch", "nobatch")
  ) |>
    mutate(reason = "oom"),
  expand_grid(
    network = c("bitcoin", "dbpedia_link", "livejournal", "microsoft_concept"),
    method = "shellstruct",
    query_type = c("multi", "single"),
    batch_type = c("batch", "nobatch")
  ) |>
    mutate(reason = "timeout"),
  expand_grid(
    network = c("friendster", "twitter_social"),
    method = "csk",
    query_type = c("single"),
    batch_type = c("batch", "nobatch")
  ) |>
    mutate(reason = "oom"),
) |>
  mutate(
    wall_s = 14400, wall_s_q1 = 14400, wall_s_q3 = 14400,
  ) |>
  select(network, method, reason, wall_s, wall_s_q1, wall_s_q3, query_type, batch_type)

cs_success <- read.csv("testing-commsearch.csv") |>
  filter(b == 1 | b == 100) |> # FIXME
  group_by(network, method, n, b) |>
  filter(all(value[stat == "exit_code"] == 0)) |>
  mutate(
    query_type = factor(if_else(n > 1, "multi", "single")),
    batch_type = factor(if_else(b > 1, "batch", "nobatch"))
  ) |>
  ungroup() |>
  filter(stat == "wall_s") |>
  group_by(network, method, query_type, batch_type) |>
  summarize(
    wall_s_q1 = quantile(value, .25),
    wall_s_q3 = quantile(value, .75),
    wall_s = mean(value),
    reason = "",
    .groups = "drop",
  ) |>
  left_join(
    read.csv("testing-offline.csv") |>
      group_by(network, method) |>
      filter(all(value[stat == "exit_code"] == 0)) |>
      ungroup() |>
      filter(stat == "wall_s") |>
      pivot_wider(names_from = method, values_from = value),
    by = "network"
  ) |>
  mutate(
    offline_add = case_when(
      method %in% c("csk", "steiner") ~ `ib-core-decomp`,
      method == "par-shellstruct" ~ `ib-core-decomp` + `par-shellstruct-offline`
    ),
    wall_s = wall_s + offline_add,
    wall_s_q1 = wall_s_q1 + offline_add,
    wall_s_q3 = wall_s_q3 + offline_add,
  ) |>
  select(network, method, reason, wall_s, wall_s_q1, wall_s_q3, query_type, batch_type)

METHODS <- c("steiner", "par-shellstruct", "csk", "shellstruct")
NETWORKS <- c("livejournal", "bitcoin", "wikipedia_link", "microsoft_concept", "dbpedia_link", "twitter_social", "friendster")

do_plot <- function(df) {
  df |>
    mutate(
      method = factor(method, levels = METHODS, labels = c("SteinerKCore", "Par-Shellstruct", "CSK", "Shellstruct")),
      network = factor(network, levels = NETWORKS)
    ) |>
    ggplot(aes(x = network, y = wall_s, fill = method)) +
    geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
    geom_errorbar(aes(ymin = wall_s_q1, ymax = wall_s_q3),
      position = position_dodge2(width = 0.9, preserve = "single")
    ) +
    geom_text(aes(label = reason),
      position = position_dodge2(width = 0.9, preserve = "single"),
      vjust = 1.3, size = 3, angle = 20, hjust = 1,
    ) +
    scale_x_discrete(name = "") +
    scale_y_continuous(name = "Runtime (s)", transform = "log10") +
    scale_fill_manual(
      name = "",
      values = c(
        "SteinerKCore" = "#F8766D",
        "Par-Shellstruct" = "#7CAE00",
        "CSK" = "#00BFC4",
        "Shellstruct" = "#C77CFF"
      )
    ) +
    theme_bw() +
    theme(
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10, angle = 20, hjust = 1),
      legend.position = "bottom",
      legend.margin = margin(t = -10, unit = "mm")
    )
}

do_plot(bind_rows(cs_success, cs_failed) |>
  filter(query_type == "single", batch_type == "nobatch"))
ggsave("testing-e1.pdf", width = 122, height = 80, units = "mm")

do_plot(bind_rows(cs_success, cs_failed) |>
  filter(query_type == "single", batch_type == "batch"))
ggsave("testing-e2.pdf", width = 122, height = 80, units = "mm")

do_plot(bind_rows(cs_success, cs_failed) |>
  filter(query_type == "multi", batch_type == "nobatch"))
ggsave("testing-e3.pdf", width = 122, height = 80, units = "mm")

do_plot(bind_rows(cs_success, cs_failed) |>
  filter(query_type == "multi", batch_type == "batch"))
ggsave("testing-e4.pdf", width = 122, height = 80, units = "mm")

# ======== runtime comparing steiner vs shellstruct (baseline of 16 thread core decomposition)

data <-
  read.csv("testing-commsearch.csv") |>
  filter(method == "par-shellstruct" | method == "steiner", stat == "wall_s") |>
  left_join(
    read.csv("testing-varythreads.csv") |>
      pivot_wider(names_from = method, values_from = time),
    by = "network", relationship = "many-to-many"
  ) |>
  mutate(
    time = if_else(
      method == "par-shellstruct",
      value + coredecomp + shellstruct,
      value + coredecomp
    )
  ) |>
  group_by(network, threads) |>
  mutate(time = time / first(coredecomp), ) |>
  ungroup() |>
  select(network, method, threads, time)

## conclusion: results are mostly robust to number of threads
## core decomposition time is pretty significant
data |>
  mutate(method = recode(
    method,
    "par-shellstruct" = "Par-Shellstruct",
    "steiner" = "SteinerKCore"
  )) |>
  group_by(network, method, threads) |>
  summarize(value = mean(time), q1 = quantile(time, .25), q3 = quantile(time, .75), .groups = "drop") |>
  ggplot(aes(x = factor(network, levels = NETWORKS), y = value, fill = method)) +
  # geom_boxplot() +
  geom_col(position = position_dodge2(width = 0.9, preserve = "single"), ) +
  geom_errorbar(aes(ymin = q1, ymax = q3),
    position = position_dodge2(width = 0.9, preserve = "single")
  ) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "darkgreen") +
  facet_grid(rows = vars(factor(threads))) +
  scale_x_discrete(name = NULL) +
  scale_y_continuous(name = "Pipeline Time (normalized)", limits = c(0, 5)) +
  theme_bw() +
  theme(
    strip.text = element_text(size = 8),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.text = element_text(size = 12),
    legend.title = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(0.4, 0.9),
    legend.direction = "horizontal",
    legend.margin = margin(t = -3, unit = "mm"),
    axis.text.x = element_text(size = 8, angle = 10, hjust = 1),
  )

ggsave("vary_threads.pdf", width = 122, height = 80, units = "mm")

# ======== runtime comparing algorithms on training networks

data <- read.csv("train-commsearch.csv")
data |> head()

## missing 18 data points for abm14 size=10 local-upper
data |>
  group_by(network, query_size, method) |>
  summarize(n = n())

df <- expand.grid(
  network = c("abm14", "cen"),
  rep = 0:19,
  query_size = c(1, 10),
  method = c("local", "local-upper", "steiner", "shell"),
  wall_s = 4 * 60 * 60
) |> rows_upsert(data, by = c("network", "rep", "query_size", "method"))

df |>
  mutate(
    method = factor(method,
      levels = c("local", "local-upper", "steiner", "shell"),
      labels = c("Local(no-ub)", "Local(yes-ub)", "SteinerKCore", "Par-Shellstruct")
    ),
  ) |>
  group_by(network, query_size, method) |>
  summarize(
    wall_s_q1 = quantile(wall_s, .25),
    wall_s_q3 = quantile(wall_s, .75),
    wall_s = mean(wall_s)
  ) |>
  ggplot(aes(x = factor(query_size), y = wall_s, fill = method)) +
  geom_col(position = position_dodge2(width = 0.9, preserve = "single"), ) +
  geom_errorbar(aes(ymin = wall_s_q1, ymax = wall_s_q3),
    position = position_dodge2(width = 0.9, preserve = "single")
  ) +
  facet_grid(rows = vars(network), scales = "free_y") +
  scale_x_discrete(name = "Query Size") +
  scale_y_continuous(name = "Runtime (s)", transform = "log10") +
  theme_bw() +
  theme(
    strip.text = element_text(size = 8),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.text = element_text(size = 8),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.margin = margin(t = -3, unit = "mm"),
    axis.text.x = element_text(size = 8),
  )

ggsave("train-commsearch.pdf", width = 122, height = 80, units = "mm")

# ==== analyzing stats of local

data <- read.csv("train-local_stats.csv")
data |> head()

data |>
  group_by(network, method, query_size) |>
  summarize(n = n())

data |>
  group_by(method, network, query_size) |>
  summarize(et_vol_gt_vol = mean(et_vol_gt_vol)) |>
  pivot_wider(names_from = method, values_from = et_vol_gt_vol) |>
  xtable() |>
  print(include.rownames = FALSE)
