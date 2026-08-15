library(dplyr)
library(tidyverse)
library(ggh4x)
library(xtable)
library(arrow)

#### ============ training core decomposition =============

data <- read_parquet("warm-train-core-decomp.parquet")
data |> head()
data |> filter(status != "ok")

### figure: 6 core decomp methods runtime

data |>
  pivot_wider(names_from = "stat", values_from = "value") |>
  mutate(
    reason = if_else(status == "ok", "", status),
    time = if_else(reason == "timeout", 4 * 60 * 60, wall_s),
    method = factor(method,
      levels = c("ib", "ucr", "gbbs", "nk", "pkc", "lbug"),
      labels = c("icebug", "ucr", "gbbs", "networkit", "pkc", "ladybugdb")
    )
  ) |>
  ggplot(aes(x = method, y = time)) +
  geom_col(fill = "#ff6666") +
  geom_text(
    y = 1,
    aes(label = reason), angle = 90, colour = "black", vjust = 0.5, hjust = 1,
  ) +
  facet_grid(cols = vars(network)) +
  geom_hline(yintercept = 4 * 60 * 60, linetype = "dashed", color = "orange") +
  theme_bw() +
  scale_x_discrete(name = "") +
  scale_y_continuous(
    name = "time (s)", transform = "log10",
    breaks = c(1, 10, 100, 1000, 4 * 60 * 60)
  ) +
  theme(
    strip.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    axis.text.x = element_text(size = 10, angle = 15, hjust = 1),
    legend.text = element_text(size = 12),
    legend.title = element_blank(),
    legend.position = "bottom",
    plot.margin = margin(b = -10, t = 5, r = 5, l = 5)
  )
ggsave("warm-train-core-decomp.pdf", width = 122, height = 80, units = "mm")

data <- read_parquet("cold-train-core-decomp.parquet")
data |> head()

data |>
  filter(stat == "exit_code", value != 0)
data |>
  filter(stat == "wall_s")

data |>
  select(network, method, stat, value) |>
  mutate(
    method = factor(method,
      levels = c("ib", "ucr", "gbbs", "nk", "pkc"),
      labels = c("Icebug", "UCR", "GBBS", "NetworKit", "PKC")
    )
  ) |>
  filter(stat == "wall_s") |>
  ggplot(aes(x = method, y = value, fill = network)) +
  # geom_col(position = "dodge") +
  geom_col(fill = "#FF6666", position = "dodge") +
  facet_grid(cols = vars(network)) +
  theme_bw() +
  scale_x_discrete(name = "") +
  scale_y_continuous(name = "Time (s)") +
  theme(
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    axis.text.x = element_text(size = 12, angle = 15, hjust = 1),
    legend.text = element_text(size = 16),
    legend.title = element_blank(),
    legend.position = "bottom",
    plot.margin = margin(b = -10, t = 5, r = 5, l = 5)
  )
ggsave("cold-train-core-decomp.pdf", width = 122, height = 80, units = "mm")


#### ============ community search =============

data <- read_parquet("commsearch.parquet")
data |> head()

online <- data |>
  filter(stage == "online") |>
  pivot_wider(names_from = stat, values_from = value)

online |> head()
online |> count(status)

## all failures from shellstruct online
online |>
  filter(status == "failed") |>
  count(experiment, network, method)

## csk oom on friendster and twitter_social
online |>
  filter(status == "oom") |>
  count(experiment, network, method)

## some local-upper times-out on abm14
online |>
  filter(status == "timeout") |>
  count(experiment, network, method)

offline <- data |>
  filter(stage != "online") |>
  pivot_wider(names_from = "stat", values_from = "value")

offline |> head()
offline |> count(status)

## shellstruct runs out of memory on 3 networks
## (oom diagnosed from logs)
offline |>
  filter(status == "failed") |>
  count(network, method, stage)

offline |>
  filter(status == "oom") |>
  count(network, method, stage)

## shellstruct times out on 4 networks
offline |>
  filter(status == "timeout") |>
  count(network, method, stage)

### figure: training comparison of our own three methods

data |>
  filter(experiment == "training") |>
  pivot_wider(names_from = stat, values_from = value) |>
  mutate(status = factor(status,
    levels = c("ok", "timeout", "oom", "failed", "absent"), ordered = TRUE
  )) |>
  group_by(network, method, size, rep) |>
  summarise(
    time = sum(if_else(stage == "online", coalesce(query_s, wall_s), wall_s)),
    status = max(status),
    .groups = "drop"
  ) |>
  mutate(time = if_else(status == "ok", time, 4 * 60 * 60)) |>
  group_by(network, method, size) |>
  summarise(
    mean_time = mean(time),
    q1 = quantile(time, 0.10),
    q3 = quantile(time, 0.90),
    n_fail = sum(status != "ok"),
    n = n(),
    worst = max(status),
    .groups = "drop"
  ) |>
  mutate(
    reason = if_else(n_fail == 0, "", str_c(
      case_match(as.character(worst), "failed" ~ "oom", .default = as.character(worst)),
      " ", n_fail, "/", n
    )),
    method = factor(method,
      levels = c("steiner", "par-shellstruct", "local", "local-upper"),
      labels = c("SteinerKCore", "Par-ShellStruct", "LocalKCore", "LocalKCore(u)")
    ),
    size = factor(size)
  ) |>
  ggplot(aes(x = size, y = mean_time, fill = method)) +
  geom_col(position = position_dodge2(width = 0.9, preserve = "single"), ) +
  geom_errorbar(aes(ymin = q1, ymax = q3),
    position = position_dodge2(width = 0.9, preserve = "single"),
  ) +
  geom_text(aes(y = 1, label = reason),
    position = position_dodge2(width = 0.9, preserve = "single"),
    angle = 90, colour = "black", vjust = 0.5, hjust = 0
  ) +
  facet_grid(rows = vars(network)) +
  geom_hline(yintercept = 4 * 60 * 60, linetype = "dashed", color = "orange") +
  theme_bw() +
  scale_x_discrete(name = "Query size") +
  scale_y_continuous(
    name = "Time (s)", transform = "log10",
    breaks = c(1, 10, 100, 1000, 4 * 60 * 60)
  ) +
  theme(
    strip.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.text = element_text(size = 8),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.box.spacing = unit(3, "pt"),
    legend.margin = margin(0, 0, 0, 0),
    axis.title.x = element_text(margin = margin(t = 2)),
    plot.margin = margin(b = 2, t = 5, r = 5, l = 5)
  )

ggsave("train-commsearch.pdf", width = 122, height = 80, units = "mm")

### figure: testing commsearch

METHODS <- c("steiner", "par-shellstruct", "csk", "shellstruct")
NETWORKS <- c("livejournal", "bitcoin", "wikipedia_link", "microsoft_concept", "dbpedia_link", "twitter_social", "friendster")

do_plot <- function(df) {
  df |>
    group_by(network, method) |>
    summarise(
      wall_s = mean(time),
      wall_s_q1 = quantile(time, 0.10),
      wall_s_q3 = quantile(time, 0.90),
      n_fail = sum(status != "ok"),
      n = n(),
      worst = max(status),
      .groups = "drop"
    ) |>
    mutate(
      reason = if_else(n_fail == 0, "", str_c(
        case_match(as.character(worst), "failed" ~ "oom", .default = as.character(worst)),
        " ", n_fail, "/", n
      )),
      method = factor(method,
        levels = METHODS,
        labels = c("SteinerKCore", "Par-ShellStruct", "CSK", "ShellStruct")
      ),
      network = factor(network, levels = NETWORKS)
    ) |>
    ggplot(aes(x = network, y = wall_s)) +
    geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
    geom_errorbar(aes(ymin = wall_s_q1, ymax = wall_s_q3),
      position = position_dodge2(width = 0.9, preserve = "single")
    ) +
    geom_text(aes(y = 1, label = reason),
      position = position_dodge2(width = 0.9, preserve = "single"),
      angle = 90, colour = "black", vjust = 0.5, hjust = 0,
    ) +
    geom_hline(yintercept = 4 * 60 * 60, linetype = "dashed", color = "orange") +
    theme_bw() +
    scale_x_discrete(name = "") +
    scale_y_continuous(
      name = "Runtime (s)", transform = "log10",
      breaks = c(1, 10, 100, 1000, 4 * 60 * 60)
    ) +
    theme(
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10, angle = 20, hjust = 1),
      legend.text = element_text(size = 8),
      legend.position = "bottom",
      legend.box.spacing = unit(3, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      axis.title.x = element_text(margin = margin(t = -10)),
      plot.margin = margin(b = 2, t = 5, r = 5, l = 5)
    )
}

testing <- data |>
  filter(experiment == "testing") |>
  pivot_wider(names_from = stat, values_from = value) |>
  mutate(status = factor(status,
    levels = c("ok", "timeout", "oom", "failed", "absent"), ordered = TRUE
  )) |>
  group_by(network, method, size, batch, rep) |>
  summarise(
    time = sum(if_else(stage == "online", coalesce(query_s, wall_s), wall_s)),
    status = max(status),
    .groups = "drop"
  ) |>
  mutate(time = if_else(status == "ok", time, 4 * 60 * 60))

testing |>
  filter(size == 1, batch == 1) |>
  do_plot()
ggsave("test-commsearch-n1-b1.pdf", width = 122, height = 80, units = "mm")

testing |>
  filter(size != 1, batch == 1) |>
  do_plot()
ggsave("test-commsearch-n>1-b1.pdf", width = 122, height = 80, units = "mm")

testing |>
  filter(size == 1, batch == 100) |>
  do_plot()
ggsave("test-commsearch-n1-b100.pdf", width = 122, height = 80, units = "mm")

testing |>
  filter(size != 1, batch == 100) |>
  do_plot()
ggsave("test-commsearch-n>1-b100.pdf", width = 122, height = 80, units = "mm")

#### ============ strong scaling =============

data <- read_parquet("strongscaling.parquet")
data |> head()

METHODS <- c("steiner", "par-shellstruct")
NETWORKS <- c(
  "livejournal", "bitcoin", "wikipedia_link", "dbpedia_link", "abm14", "cen",
  "microsoft_concept", "twitter_social", "friendster"
)
SHAPES <- c(15, 16, 17, 18, 3, 4, 8, 0, 1)
THREADS <- c(1, 2, 4, 8, 16, 32, 48, 64)

scaling <- read_parquet("strongscaling.parquet") |>
  pivot_wider(names_from = stat, values_from = value) |>
  mutate(status = factor(status,
    levels = c("ok", "timeout", "oom", "failed", "absent"), ordered = TRUE
  )) |>
  group_by(network, method, threads, rep) |>
  summarise(
    time = sum(if_else(stage == "online", coalesce(query_s, wall_s), wall_s)),
    rss = max(peak_rss_kb, na.rm = TRUE) / 1024^2,
    status = max(status),
    .groups = "drop"
  ) |>
  mutate(time = if_else(status == "ok", time, 4 * 60 * 60))

do_plot <- function(df, y, ylab, lb) {
  df |>
    group_by(network, method, threads) |>
    summarise(
      mean = mean({{ y }}, na.rm = TRUE),
      q1 = quantile({{ y }}, 0.10, na.rm = TRUE),
      q3 = quantile({{ y }}, 0.90, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      method = factor(method,
        levels = METHODS,
        labels = c("SteinerKCore", "Par-ShellStruct")
      ),
      network = factor(network, levels = NETWORKS)
    ) |>
    ggplot(aes(
      x = threads, y = mean, shape = network,
      group = interaction(network, method)
    )) +
    geom_line(linewidth = 0.4) +
    # geom_errorbar(aes(ymin = q1, ymax = q3), width = 0.06, linewidth = 0.3) +
    geom_point(size = 1.8) +
    facet_wrap(vars(method)) +
    theme_bw() +
    scale_x_continuous(name = "Number of CPUs", transform = "log2", breaks = THREADS) +
    scale_y_continuous(name = ylab, transform = "log10", limits = c(lb, NA)) +
    scale_shape_manual(name = "", values = SHAPES) +
    guides(shape = guide_legend(order = 2, nrow = 2)) +
    theme(
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 8),
      legend.text = element_text(size = 8, margin = margin(l = 1, r = 4)),
      legend.position = "bottom",
      legend.box = "vertical",
      legend.box.spacing = unit(3, "pt"),
      legend.key.size = unit(8, "pt"),
      legend.key.spacing.x = unit(0, "pt"),
      legend.key.spacing.y = unit(0, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      axis.title.x = element_text(margin = margin(t = 0)),
      plot.margin = margin(b = 2, t = 5, r = 5, l = 5)
    )
}

scaling |> do_plot(time, "Runtime (s)", 5)
ggsave("strongscaling-line-time.pdf", width = 122, height = 80, units = "mm")

scaling |> do_plot(rss, "Peak RSS (GB)", 0.8)
ggsave("strongscaling-line-mem.pdf", width = 122, height = 80, units = "mm")
