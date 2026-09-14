library(dplyr)
library(tidyverse)
library(ggh4x)
library(xtable)
library(arrow)
library(patchwork)

TIMEOUT_S <- 4 * 60 * 60

as_status <- function(x) {
  factor(x, levels = c("ok", "timeout", "oom", "failed", "absent"), ordered = TRUE)
}

fail_label <- function(n_fail, n, worst) {
  if_else(n_fail == 0, "", str_c(as.character(worst), " ", n_fail, "/", n))
}

#### ============ training experiment =============

TRAIN_NETWORKS <- c("cen", "abm14")

core_decomp <- read_parquet("cold-train-core-decomp.parquet")
core_decomp |> head()

core_decomp |>
  filter(stat == "exit_code", value != 0)
core_decomp |>
  filter(stat == "wall_s")

core_decomp |>
  select(network, method, stat, value) |>
  filter(stat == "wall_s") |>
  mutate(
    method = factor(method,
      levels = c("ib", "ucr", "gbbs", "nk", "pkc"),
      labels = c("Icebug", "UCR", "GBBS", "NK", "PKC")
    ),
    network = factor(network, levels = TRAIN_NETWORKS)
  ) |>
  ggplot(aes(x = method, y = value)) +
  geom_col(fill = "grey50", position = "dodge") +
  facet_wrap(. ~ network) +
  theme_bw() +
  scale_x_discrete(name = "Method") +
  scale_y_continuous(name = "Time (s)") +
  theme(
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    axis.text.x = element_text(size = 8),
    axis.title.x = element_text(margin = margin(t = 2)),
    plot.margin = margin(b = 2, t = 5, r = 5, l = 5)
  )
ggsave("train-core-decomp.pdf", width = 122, height = 40, units = "mm")

train_commsearch <- read_parquet("commsearch.parquet") |> filter(experiment == "training")
train_commsearch |> head()

METHOD_COLORS <- c(
  "SteinerKCore" = "#F8766D",
  "Par-ShellStruct" = "#7CAE00",
  "LocalKCore" = "#00BFC4",
  "LocalKCore(u)" = "#C77CFF",
  "CSK" = "#00A9FF",
  "ShellStruct" = "#FF61CC"
)

train_commsearch |>
  filter(method != "par-shellstruct") |>
  pivot_wider(names_from = stat, values_from = value) |>
  mutate(status = as_status(status)) |>
  group_by(network, method, size, rep) |>
  summarise(
    time = sum(if_else(stage == "online", coalesce(query_s, wall_s), wall_s)),
    status = max(status),
    .groups = "drop"
  ) |>
  mutate(time = if_else(status == "ok", time, TIMEOUT_S)) |>
  group_by(network, method, size) |>
  summarise(
    mean_time = mean(time),
    q1 = quantile(time, 0.10),
    q3 = quantile(time, 0.90),
    n_fail = sum(status != "ok"),
    n = n(),
    worst = max(status),
    sd = sd(time),
    .groups = "drop"
  ) |>
  mutate(
    reason = fail_label(n_fail, n, worst),
    method = factor(method,
      levels = c("steiner", "par-shellstruct", "local", "local-upper"),
      labels = c("SteinerKCore", "Par-ShellStruct", "LocalKCore", "LocalKCore(u)")
    ),
    network = factor(network, levels = TRAIN_NETWORKS),
    size = factor(size)
  ) |>
  ggplot(aes(x = size, y = mean_time, fill = method)) +
  geom_col(position = position_dodge2(width = 0.9, preserve = "single"), ) +
  geom_errorbar(aes(ymin = pmax(mean_time - sd, 0), ymax = mean_time + sd),
    position = position_dodge2(width = 0.9, preserve = "single"),
  ) +
  geom_text(aes(y = 1, label = reason),
    position = position_dodge2(width = 0.9, preserve = "single"),
    angle = 90, colour = "black", vjust = 0.5, hjust = 0, size = 2.5
  ) +
  facet_wrap(. ~ network, labeller = labeller(network = toupper)) +
  geom_hline(yintercept = TIMEOUT_S, linetype = "dashed", color = "orange") +
  theme_bw() +
  scale_x_discrete(name = "Query size") +
  coord_transform(y = "log10", ylim = c(1, TIMEOUT_S)) +
  scale_y_continuous(
    name = "Time (s)",
    breaks = c(1, 10, 100, 1000, TIMEOUT_S),
    labels = c("1", "10", "100", "1000", "14400")
  ) +
  scale_fill_manual(name = "", values = METHOD_COLORS) +
  theme(
    strip.text = element_text(size = 10),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    axis.text.x = element_text(size = 8),
    legend.text = element_text(size = 9),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.key.size = unit(10, "pt"),
    legend.box.spacing = unit(3, "pt"),
    legend.margin = margin(0, 0, 0, 0),
    axis.title.x = element_text(margin = margin(t = 2)),
    plot.margin = margin(b = 2, t = 5, r = 5, l = 5)
  )

ggsave("train-commsearch.pdf", width = 122, height = 50, units = "mm")


#### ============ testing community search =============

data <- read_parquet("commsearch.parquet")
data |> head()
data |> count(network)

online <- data |>
  filter(stage == "online") |>
  pivot_wider(names_from = stat, values_from = value)

online |> head()
online |> count(status)

## failures from shellstruct online, and from csk on friendster and twitter_social
online |>
  filter(status == "failed") |>
  count(experiment, network, method)

## csk oom on abm272, friendster and twitter_social
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

## shellstruct runs out of memory on 4 networks
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

### figure: testing commsearch

METHODS <- c("steiner", "par-shellstruct", "csk", "shellstruct")
NETWORKS <- c("livejournal", "bitcoin", "wikipedia_link", "microsoft_concept", "dbpedia_link", "twitter_social", "friendster", "abm272")

testing <- function(stages) {
  data |>
    filter(experiment == "testing", method %in% METHODS, stage %in% stages) |>
    pivot_wider(names_from = stat, values_from = value) |>
    mutate(status = as_status(status)) |>
    group_by(network, method, size, batch, rep) |>
    summarise(
      time = sum(if_else(stage == "online", coalesce(query_s, wall_s), wall_s)),
      status = if (any(stage != "online" & status != "ok")) {
        max(status[stage != "online"])
      } else {
        max(status)
      },
      .groups = "drop"
    ) |>
    mutate(time = if_else(status == "ok", time, TIMEOUT_S))
}

NETWORK_LABELS <- c(
  livejournal = "LiveJournal", bitcoin = "Bitcoin", wikipedia_link = "Wikipedia",
  microsoft_concept = "MS-Concept", dbpedia_link = "DBpedia", twitter_social = "Twitter",
  friendster = "Friendster", abm272 = "ABM272", abm14 = "ABM14", cen = "CEN"
)

testing(c("offline", "online")) |>
  filter(size %in% c(1, 20), batch != 5) |>
  group_by(network, method, size, batch) |>
  summarise(
    wall_s = mean(time),
    n_fail = sum(status != "ok"),
    worst = max(status),
    sd = sd(time),
    .groups = "drop"
  ) |>
  mutate(
    worst = case_match(as.character(worst), "failed" ~ "oom", .default = as.character(worst)),
    reason = if_else(n_fail == 0, "", worst),
    method = factor(method,
      levels = METHODS,
      labels = c("SteinerKCore", "Par-ShellStruct", "CSK", "ShellStruct")
    ),
    network = factor(network, levels = NETWORKS, labels = NETWORK_LABELS[NETWORKS]),
    batch = factor(batch)
  ) |>
  droplevels() |>
  complete(network, method, size, batch, fill = list(reason = "")) |>
  ggplot(aes(x = batch, y = wall_s, fill = method)) +
  geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  geom_errorbar(aes(ymin = pmax(wall_s - sd, 0), ymax = wall_s + sd),
    position = position_dodge2(width = 0.9, preserve = "single")
  ) +
  geom_text(aes(y = wall_s, label = reason),
    position = position_dodge2(width = 0.9, preserve = "single"),
    angle = 90, colour = "black", vjust = 0.5, hjust = 1.05, size = 2.5
  ) +
  geom_hline(yintercept = TIMEOUT_S, linetype = "dashed", color = "orange") +
  facet_grid(
    rows = vars(network), cols = vars(size),
    labeller = labeller(size = function(x) str_c("n = ", x))
  ) +
  theme_bw() +
  scale_x_discrete(name = "Number of queries") +
  coord_transform(y = "log10", ylim = c(0.1, TIMEOUT_S)) +
  scale_y_continuous(
    name = "Runtime (s)",
    breaks = c(0.1, 1, 10, 100, 1000, TIMEOUT_S),
    labels = c("0.1", "1", "10", "100", "1000", "14400")
  ) +
  scale_fill_manual(name = "", values = METHOD_COLORS) +
  theme(
    strip.text = element_text(size = 8),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 8),
    legend.text = element_text(size = 8),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.box.spacing = unit(3, "pt"),
    legend.margin = margin(0, 0, 0, 0),
    axis.title.x = element_text(margin = margin(t = 2)),
    plot.margin = margin(b = 2, t = 5, r = 5, l = 5)
  )
ggsave("test-commsearch.pdf", width = 122, height = 160, units = "mm")

testing(c("offline", "online")) |>
  filter(size %in% c(5, 10), batch != 5) |>
  group_by(network, method, size, batch) |>
  summarise(
    wall_s = mean(time),
    n_fail = sum(status != "ok"),
    worst = max(status),
    sd = sd(time),
    .groups = "drop"
  ) |>
  mutate(
    worst = case_match(as.character(worst), "failed" ~ "oom", .default = as.character(worst)),
    reason = if_else(n_fail == 0, "", worst),
    method = factor(method,
      levels = METHODS,
      labels = c("SteinerKCore", "Par-ShellStruct", "CSK", "ShellStruct")
    ),
    network = factor(network, levels = NETWORKS, labels = NETWORK_LABELS[NETWORKS]),
    batch = factor(batch)
  ) |>
  droplevels() |>
  complete(network, method, size, batch, fill = list(reason = "")) |>
  ggplot(aes(x = batch, y = wall_s, fill = method)) +
  geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  geom_errorbar(aes(ymin = pmax(wall_s - sd, 0), ymax = wall_s + sd),
    position = position_dodge2(width = 0.9, preserve = "single")
  ) +
  geom_text(aes(y = wall_s, label = reason),
    position = position_dodge2(width = 0.9, preserve = "single"),
    angle = 90, colour = "black", vjust = 0.5, hjust = 1.05, size = 2.5
  ) +
  geom_hline(yintercept = TIMEOUT_S, linetype = "dashed", color = "orange") +
  facet_grid(
    rows = vars(network), cols = vars(size),
    labeller = labeller(size = function(x) str_c("n = ", x))
  ) +
  theme_bw() +
  scale_x_discrete(name = "Number of queries") +
  coord_transform(y = "log10", ylim = c(0.1, TIMEOUT_S)) +
  scale_y_continuous(
    name = "Runtime (s)",
    breaks = c(0.1, 1, 10, 100, 1000, TIMEOUT_S),
    labels = c("0.1", "1", "10", "100", "1000", "14400")
  ) +
  scale_fill_manual(name = "", values = METHOD_COLORS) +
  theme(
    strip.text = element_text(size = 8),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 8),
    legend.text = element_text(size = 8),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.box.spacing = unit(3, "pt"),
    legend.margin = margin(0, 0, 0, 0),
    axis.title.x = element_text(margin = margin(t = 2)),
    plot.margin = margin(b = 2, t = 5, r = 5, l = 5)
  )
ggsave("test-commsearch-n5-n10.pdf", width = 122, height = 160, units = "mm")

testing(c("offline", "online")) |>
  filter(
    size == 1,
    method %in% c("csk", "steiner"),
    !network %in% c("friendster", "twitter_social")
  ) |>
  mutate(time = if_else(status == "ok", time, TIMEOUT_S)) |>
  group_by(network, batch, rep) |>
  summarize(speedup = sum(time[method == "csk"]) / sum(time[method == "steiner"])) |>
  group_by(network, batch) |>
  summarize(speedup = median(speedup)) |>
  pivot_wider(names_from = batch, values_from = speedup)

#### ============ strong scaling =============

data <- read_parquet("strongscaling.parquet")
data |> head()

data |>
  pivot_wider(names_from = stat, values_from = value) |>
  group_by(network, method, threads, rep) |>
  summarize(
    time = sum(if_else(stage == "online", coalesce(query_s, wall_s), wall_s)),
    status = max(status),
    .groups = "drop"
  ) |>
  mutate(time = if_else(status == "ok", time, TIMEOUT_S)) |>
  group_by(network, method, threads) |>
  summarize(time = mean(time), .groups = "drop") |>
  group_by(network, method) |>
  mutate(speedup = time[threads == 1] / time) |>
  ungroup() |>
  select(network, method, threads, speedup) |>
  pivot_wider(names_from = threads, values_from = speedup)

METHODS <- c("steiner", "par-shellstruct")
NETWORKS <- c(
  "livejournal", "bitcoin", "wikipedia_link", "dbpedia_link", "abm14", "cen",
  "microsoft_concept", "twitter_social", "friendster", "abm272"
)
SCALING_NETWORKS <- c(bitcoin = "Small", wikipedia_link = "Medium", friendster = "Large")
SHAPES <- c(15, 16, 17)
ALL_SHAPES <- c(15, 16, 17, 18, 3, 4, 8, 0, 1)
THREADS <- c(1, 2, 4, 8, 16, 32, 48, 64)

scaling <- read_parquet("strongscaling.parquet") |>
  pivot_wider(names_from = stat, values_from = value) |>
  mutate(status = as_status(status)) |>
  group_by(network, method, threads, rep) |>
  summarise(
    time = sum(if_else(stage == "online", coalesce(query_s, wall_s), wall_s)),
    status = max(status),
    .groups = "drop"
  ) |>
  mutate(time = if_else(status == "ok", time, TIMEOUT_S))

scaling |>
  filter(network %in% names(SCALING_NETWORKS)) |>
  group_by(network, method, threads) |>
  summarise(
    mean = mean(time, na.rm = TRUE),
    q1 = quantile(time, 0.10, na.rm = TRUE),
    q3 = quantile(time, 0.90, na.rm = TRUE),
    se = sd(time, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) |>
  mutate(
    method = factor(method,
      levels = METHODS,
      labels = c("SteinerKCore", "Par-ShellStruct")
    ),
    network = factor(network,
      levels = names(SCALING_NETWORKS), labels = SCALING_NETWORKS
    )
  ) |>
  ggplot(aes(
    x = threads, y = mean, shape = network,
    group = interaction(network, method)
  )) +
  geom_line(linewidth = 0.4) +
  # geom_errorbar(aes(ymin = mean - 2 * se, ymax = mean + 2 * se), width = 0.06, linewidth = 0.3) +
  geom_point(size = 1.8) +
  facet_wrap(vars(method)) +
  theme_bw() +
  scale_x_continuous(name = "Number of CPUs", transform = "log2", breaks = THREADS) +
  scale_y_continuous(name = "Runtime (s)", transform = "log10", limits = c(5, NA)) +
  scale_shape_manual(name = "", values = SHAPES) +
  guides(shape = guide_legend(order = 2, reverse = TRUE)) +
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 8),
    legend.text = element_text(size = 8),
    legend.title = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(0.985, 0.97),
    legend.justification = c(1, 1),
    legend.background = element_rect(fill = "white", colour = NA),
    legend.key.size = unit(10, "pt"),
    legend.margin = margin(2, 4, 2, 2),
    axis.title.x = element_text(margin = margin(t = 0)),
    plot.margin = margin(b = 2, t = 5, r = 5, l = 5)
  )

ggsave("strongscaling.pdf", width = 122, height = 60, units = "mm")

scaling |>
  group_by(network, method, threads) |>
  summarise(
    mean = mean(time, na.rm = TRUE),
    q1 = quantile(time, 0.10, na.rm = TRUE),
    q3 = quantile(time, 0.90, na.rm = TRUE),
    se = sd(time, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) |>
  mutate(
    method = factor(method,
      levels = METHODS,
      labels = c("SteinerKCore", "Par-ShellStruct")
    ),
    network = factor(network, levels = NETWORKS, labels = NETWORK_LABELS[NETWORKS])
  ) |>
  ggplot(aes(
    x = threads, y = mean, shape = network,
    group = interaction(network, method)
  )) +
  geom_line(linewidth = 0.4) +
  # geom_errorbar(aes(ymin = mean - 2 * se, ymax = mean + 2 * se), width = 0.06, linewidth = 0.3) +
  geom_point(size = 1.8) +
  facet_wrap(vars(method)) +
  theme_bw() +
  scale_x_continuous(name = "Number of CPUs", transform = "log2", breaks = THREADS) +
  scale_y_continuous(name = "Runtime (s)", transform = "log10", limits = c(5, NA)) +
  scale_shape_manual(name = "", values = ALL_SHAPES) +
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

ggsave("strongscaling-all.pdf", width = 122, height = 80, units = "mm")

#### ============ cold, warm, simult =============

data <- read_parquet("cold-warm.parquet")
data |> head()

## everything is okay
data |>
  pivot_wider(names_from = stat, values_from = value) |>
  filter(status != "ok")

data |>
  mutate(
    cache = recode(cache, cold = "Cold Cache", warm = "Warm Cache"),
    mode = recode(mode, serial = "No Co-Scheduling", simult = "Co-Scheduling")
  ) |>
  pivot_wider(names_from = stat, values_from = value) |>
  group_by(network, mode, cache) |>
  summarize(wall_s = mean(wall_s)) |>
  ggplot(aes(x = cache, y = wall_s, fill = mode)) +
  geom_col(position = "dodge") +
  facet_grid(rows = vars(network), scales = "free_y") +
  scale_color_manual(name = "") +
  scale_x_discrete(name = "") +
  scale_y_continuous(name = "Runtime (s)") +
  theme_bw() +
  theme(
    strip.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    legend.text = element_text(size = 12),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.margin = margin(0, 0, 0, 0),
    plot.margin = margin(b = 2, t = 5, r = 5, l = 5),
    legend.box.spacing = unit(-10, "pt"),
  )
ggsave("cold-warm.pdf", width = 122, height = 60, units = "mm")

#### ============ query analysis =============

data <- read_parquet("query-analysis.parquet")
data |> head()

## no non-ok datapoints
data |> filter(status != "ok")

mark_best <- function(df, cols, digits = 1, best = c("max", "min")) {
  best <- match.arg(best)
  df <- ungroup(df)
  picked <- names(tidyselect::eval_select(rlang::enquo(cols), df))

  rank_one <- function(x) {
    r <- rank(if (best == "max") -x else x, ties.method = "min", na.last = "keep")
    s <- formatC(x, format = "d", digits = digits)
    s[which(r == 1)] <- paste0("\\textbf{", s[which(r == 1)], "}")
    s[which(r == 2)] <- paste0("\\underline{", s[which(r == 2)], "}")
    s
  }

  df |>
    mutate(.row = row_number()) |>
    pivot_longer(all_of(picked), names_to = ".metric", values_to = ".value") |>
    group_by(.row) |>
    mutate(.value = rank_one(.value)) |>
    ungroup() |>
    pivot_wider(names_from = ".metric", values_from = ".value") |>
    select(all_of(names(df)))
}

### table: threshold==0.99 query coreness, restricted networks

data |>
  filter(network %in% c("livejournal", "friendster"), threshold == 0.99) |>
  group_by(network, centrality, threshold) |>
  summarize(cores = median(cores)) |>
  select(network, centrality, cores) |>
  mutate(centrality = factor(centrality,
    levels = c("coreness", "degree", "pagerank", "c_coef"),
    labels = c("Coreness", "Degree", "PageRank", "Clustering Coef.")
  )) |>
  arrange(centrality) |>
  pivot_wider(names_from = centrality, values_from = cores) |>
  mark_best(c(`Clustering Coef.`, `Degree`, `PageRank`)) |>
  xtable(digits = 0, align = c("l", "l", "r", "r", "r", "r")) |>
  print(include.rownames = FALSE, sanitize.text.function = identity)

data |>
  filter(threshold == 0.99) |>
  group_by(network, centrality, threshold) |>
  summarize(cores = median(cores)) |>
  select(network, centrality, cores) |>
  mutate(centrality = factor(centrality,
    levels = c("coreness", "degree", "pagerank", "c_coef"),
    labels = c("Coreness", "Degree", "PageRank", "Clustering Coef.")
  )) |>
  arrange(centrality) |>
  pivot_wider(names_from = centrality, values_from = cores) |>
  mark_best(c(`Clustering Coef.`, `Degree`, `PageRank`)) |>
  xtable(digits = 0, align = c("l", "l", "r", "r", "r", "r")) |>
  print(include.rownames = FALSE, sanitize.text.function = identity)

data |>
  filter(threshold == 0.999) |>
  group_by(network, centrality, threshold) |>
  summarize(cores = median(cores)) |>
  select(network, centrality, cores) |>
  mutate(centrality = factor(centrality,
    levels = c("coreness", "degree", "pagerank", "c_coef"),
    labels = c("Coreness", "Degree", "PageRank", "Clustering Coef.")
  )) |>
  arrange(centrality) |>
  pivot_wider(names_from = centrality, values_from = cores) |>
  mark_best(c(`Clustering Coef.`, `Degree`, `PageRank`)) |>
  xtable(digits = 0, align = c("l", "l", "r", "r", "r", "r")) |>
  print(include.rownames = FALSE, sanitize.text.function = identity)

### table: network stats

data <- read_parquet("network-stats.parquet")
data |> head()

## all runs are ok
data |> filter(status != "ok")

data |>
  select(network, stat, value) |>
  pivot_wider(names_from = stat, values_from = value)

#### comparing format time and core decomp time

csr_format <- read_parquet("csr-format.parquet")
csr_format |> head()

core_decomp <- read_parquet("commsearch.parquet") |>
  filter(stage == "core-decomp") |>
  group_by(network, stage, stat, status) |>
  summarize(value = first(value)) |>
  ungroup()
core_decomp |> head()

rbind(core_decomp, csr_format) |>
  filter(stat == "wall_s", stage %in% c("core-decomp", "csv2csr")) |>
  select(network, stage, value) |>
  pivot_wider(names_from = "stage", values_from = value) |>
  mutate(
    network = factor(network, levels = NETWORKS),
    ratio = csv2csr / `core-decomp`
  ) |>
  arrange(network)

#### ============ scaling on abm272 =============

data <- read_parquet("abm272.parquet") |> filter(status == "ok")
data |> head()
data |> count(stage)
data |> count(stat)

data |>
  select(nodes) |>
  min()
data |>
  select(nodes) |>
  max()

plot_data <- data |>
  filter(stage %in% c("core-decomp", "shellstruct-offline", "shellstruct-online", "steiner")) |>
  mutate(want = if_else(stage %in% c("shellstruct-online", "steiner"), "query_s", "wall_s")) |>
  filter(stat == want | stat == "peak_rss_kb") |>
  mutate(metric = if_else(stat == "peak_rss_kb", "rss", "time")) |>
  group_by(year, nodes, stage, metric) |>
  summarize(value = mean(value), .groups = "drop") |>
  pivot_wider(names_from = "stage", values_from = "value") |>
  mutate(
    shellstruct = if_else(metric == "time",
      `core-decomp` + `shellstruct-offline` + `shellstruct-online`,
      pmax(`core-decomp`, `shellstruct-offline`, `shellstruct-online`) / 1024^2
    ),
    steiner = if_else(metric == "time",
      `core-decomp` + steiner,
      pmax(`core-decomp`, steiner) / 1024^2
    )
  ) |>
  select(year, nodes, metric, shellstruct, steiner) |>
  pivot_longer(c(shellstruct, steiner), names_to = "method", values_to = "value") |>
  mutate(
    metric = factor(metric,
      levels = c("time", "rss"),
      labels = c("Runtime (s)", "Memory (GB)")
    ),
    color = if_else(method == "shellstruct", "#7CAE00", "#F8766D")
  )

node_labels <- tibble(
  metric = factor(
    rep(c("Runtime (s)", "Memory (GB)"), each = 2),
    levels = c("Runtime (s)", "Memory (GB)")
  ),
  x = rep(c(log10(26598), log10(272739486)), 2),
  y = rep(c(10, 0.94), each = 2),
  label = rep(c("26,598", "272,739,486"), 2)
)

plot_data |> ggplot(aes(x = log10(nodes), y = value, color = color)) +
  geom_line() +
  geom_point() +
  geom_vline(
    xintercept = c(log10(26598), log10(272739486)), linetype = "dashed",
    color = "grey40", linewidth = 0.3
  ) +
  geom_text(
    data = node_labels, aes(x = x, y = y, label = label), inherit.aes = FALSE,
    hjust = -0.1, vjust = 1, size = 3, angle = 90, color = "grey30"
  ) +
  facet_grid(rows = vars(metric), scales = "free_y") +
  facetted_pos_scales(y = list(
    metric == "Runtime (s)" ~ scale_y_log10(name = NULL, limits = c(1, TIMEOUT_S)),
    metric == "Memory (GB)" ~ scale_y_log10(name = NULL)
  )) +
  scale_x_continuous(
    name = "log10(# Nodes)", limits = c(log10(26598), log10(272739486)),
    breaks = c(5, 6, 7, 8),
  ) +
  scale_color_identity(
    guide = "legend",
    breaks = c("#7CAE00", "#F8766D"),
    labels = c("Par-ShellStruct", "SteinerKCore")
  ) +
  theme_bw() +
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    strip.text = element_text(size = 10),
    legend.text = element_text(size = 8),
    legend.title = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(0.015, 0.97),
    legend.justification = c(0, 1),
    legend.background = element_rect(fill = "white", colour = NA),
    legend.key.size = unit(10, "pt"),
    legend.margin = margin(2, 4, 2, 2),
    axis.title.x = element_text(margin = margin(t = 0)),
    plot.margin = margin(b = 2, t = 5, r = 5, l = 5)
  )

ggsave("abm272-commsearch-scaling.pdf", width = 122, height = 60, units = "mm")

data |>
  mutate(want = if_else(stage %in% c("shellstruct-online", "steiner"), "query_s", "wall_s")) |>
  filter(
    year == 2136,
    stage %in% c("core-decomp", "shellstruct-offline", "shellstruct-online", "steiner"),
    stat == want
  ) |>
  group_by(stage) |>
  summarize(value = mean(value))


#### ============ power-law exponents on abm272 =============

data <- read_parquet("abm272.parquet") |> filter(status == "ok")

pipelines <- data |>
  filter(stage %in% c("core-decomp", "steiner", "shellstruct-offline", "shellstruct-online")) |>
  mutate(want = if_else(stage %in% c("steiner", "shellstruct-online"), "query_s", "wall_s")) |>
  filter(stat == want | stat == "peak_rss_kb") |>
  mutate(metric = if_else(stat == "peak_rss_kb", "mem_gb", "time_s")) |>
  group_by(metric, year, nodes, stage) |>
  summarize(value = mean(value), .groups = "drop") |>
  pivot_wider(names_from = "stage", values_from = "value") |>
  mutate(
    shellstruct = if_else(metric == "time_s",
      `core-decomp` + `shellstruct-offline` + `shellstruct-online`,
      pmax(`core-decomp`, `shellstruct-offline`, `shellstruct-online`)
    ),
    steiner = if_else(metric == "time_s",
      `core-decomp` + steiner,
      pmax(`core-decomp`, steiner)
    )
  ) |>
  select(metric, year, nodes, shellstruct, steiner) |>
  pivot_longer(c(shellstruct, steiner), names_to = "pipeline", values_to = "value") |>
  mutate(value = if_else(metric == "mem_gb", value / 1e6, value))

slope <- function(df) {
  fit <- lm(log10(value) ~ log10(nodes), data = df)
  tibble(
    alpha = round(coef(fit)[["log10(nodes)"]], 2),
    r2 = round(summary(fit)$r.squared, 2)
  )
}

pipelines |>
  filter(metric == "mem_gb", nodes >= 1e6) |>
  group_by(pipeline) |>
  group_modify(~ slope(.x)) |>
  print()

pipelines |>
  filter(metric == "time_s") |>
  mutate(range = if_else(nodes <= 5e7, "<=5e7", ">5e7")) |>
  group_by(pipeline, range) |>
  group_modify(~ slope(.x)) |>
  print()

# ===

data <- read_parquet("abm-sizes.parquet")
data |> head()
data |>
  filter(year == 2146) |>
  mutate(kavg = 2 * edges / nodes)
