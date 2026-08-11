library(dplyr)
library(tidyverse)
library(ggh4x)
library(xtable)

# ================= QUERY ANALYSISS ===============

data <- rbind(
  read.csv("cen.query_analysis.csv"),
  read.csv("abm14.query_analysis.csv")
)
head(data)

### sampling according to degree is best
data |>
  filter(
    size %in% c(1, 10, 25),
    threshold %in% c(0.9, 0.99, 0.999),
    network == "abm14",
    centrality != "coreness",
  ) |>
  group_by(network, size, threshold, centrality) |>
  summarize(cores = quantile(cores, 0.99)) |>
  pivot_wider(names_from = "centrality", values_from = "cores") |>
  xtable() |>
  print(include.rownames = FALSE)

### sampling according to degree is best
data |>
  filter(
    size %in% c(1, 10, 25),
    threshold %in% c(0.9, 0.99, 0.999),
    network == "cen",
    centrality != "coreness",
  ) |>
  group_by(network, size, threshold, centrality) |>
  summarize(cores = quantile(cores, 0.99)) |>
  pivot_wider(names_from = "centrality", values_from = "cores") |>
  xtable() |>
  print(include.rownames = FALSE)

data <- read.csv("cen.query_analysis.csv")
head(data)
lapply(data, unique)

### larger query sets => smaller coreness values
### lower threshold   => smaller coreness values
### not too much visible difference across centrality measures
data |>
  mutate(
    threshold = factor(threshold,
      levels = c("0.9999", "0.999", "0.99", "0.9", "0"),
      labels = c(">0.9999", ">0.999", ">0.99", ">0.9", ">0.0"),
    ),
    centrality = factor(centrality,
      levels = c("coreness", "degree", "pagerank", "c_coef")
    ),
  ) |>
  ggplot(aes(x = factor(size), y = cores)) +
  facet_grid(
    rows = vars(threshold),
    cols = vars(centrality),
    scales = "fixed"
  ) +
  geom_boxplot() +
  theme_bw() +
  scale_x_discrete(name = "vertices per query") +
  scale_y_continuous(limits = c(0, 60)) +
  theme(
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14, angle = 60, hjust = 1)
  )
ggsave("cen.query_analysis.pdf", dpi = 300)

data <- read.csv("abm14.query_analysis.csv")
head(data)
lapply(data, unique)

data |>
  mutate(
    threshold = factor(threshold,
      levels = c("0.9999", "0.999", "0.99", "0.9", "0"),
      labels = c(">0.9999", ">0.999", ">0.99", ">0.9", ">0.0"),
    ),
    centrality = factor(centrality,
      levels = c("coreness", "degree", "pagerank", "c_coef")
    ),
  ) |>
  ggplot(aes(x = factor(size), y = cores)) +
  facet_grid(
    rows = vars(threshold),
    cols = vars(centrality),
    scales = "fixed"
  ) +
  geom_boxplot() +
  theme_bw() +
  scale_x_discrete(name = "vertices per query") +
  scale_y_continuous(limits = c(0, 120)) +
  theme(
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14, angle = 60, hjust = 1)
  )
ggsave("abm14.query_analysis.pdf", dpi = 300)

# ================= SHELL ANALYSIS ===============

data <-
  rbind(
    read.csv("abm14.coresizes.csv") |> mutate(network = "abm14"),
    read.csv("bitcoin.coresizes.csv") |> mutate(network = "bitcoin"),
    read.csv("cen.coresizes.csv") |> mutate(network = "cen"),
    read.csv("dbpedia_link.coresizes.csv") |> mutate(network = "dbpedia_link"),
    read.csv("friendster.coresizes.csv") |> mutate(network = "friendster"),
    read.csv("livejournal.coresizes.csv") |> mutate(network = "livejournal"),
    read.csv("microsoft_concept.coresizes.csv") |> mutate(network = "microsoft_concept"),
    read.csv("twitter_social.coresizes.csv") |> mutate(network = "twitter_social"),
    read.csv("wikipedia_link.coresizes.csv") |> mutate(network = "wikipedia_link")
  )
head(data)
lapply(data, unique)

data |>
  group_by(network) |>
  summarize(n = n())

data |>
  ggplot(aes(x = core, y = sizes, color = network)) +
  geom_point() +
  scale_y_continuous(transform = "log10") +
  theme_bw()

## there are not that many distinct maximal connected cores
## except on microsoft, there is the most by far
data |>
  group_by(network) |>
  summarize(n = n())

## there are no "bridges", i.e. disconnected k-cores for nontrivial k
## except for livejournal
data |>
  group_by(network, core) |>
  summarize(ncomp = n()) |>
  filter(ncomp > 1) |>
  group_by(network) |>
  summarize(n = n())

data |>
  group_by(network) |>
  summarize(degeneracy = max(core))

## warm-cold trials

data <- read.csv("cold-warm.csv") |>
  mutate(env = factor(env, levels = c("warm", "cold", "simult")))
data |> head()
lapply(data, unique)

data |>
  filter(key %in% c("runtime", "wall_s")) |>
  mutate(key = case_match(
    key,
    "runtime" ~ "no I/O",
    "wall_s" ~ "with I/O",
  )) |>
  ggplot(aes(x = env, y = value, fill = key)) +
  geom_boxplot() +
  facet_grid(
    rows = vars(method),
  ) +
  theme_bw() +
  scale_x_discrete(name = "Environment") +
  scale_y_continuous(name = "Time (s)", limits = c(0, NA)) +
  theme(
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14, angle = 60, hjust = 1)
  )

ggsave("cold-warm-simult.pdf", dpi = 300)

## core decomposition

data <- read.csv("train-core-decomp.csv")
data |> head()

data |>
  filter(stat == "exit_code", value != 0)

lapply(data |> select(-c("value")), table)

data |>
  select(network, method, stat, value) |>
  filter(stat %in% c("wall_s", "peak_rss_kb")) |>
  mutate(value = if_else(stat == "peak_rss_kb", value / 1e6, value)) |>
  pivot_wider(names_from = "stat", values_from = "value") |>
  rename(`Time (s)` = wall_s, `Memory (GB)` = peak_rss_kb) |>
  xtable() |>
  print(include.rownames = FALSE)

## icebug shellstruct strong scaling

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
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.title = element_blank(),
    legend.position = "bottom"
  )
ggsave("shellstruct_strong_scaling.pdf")

## icebug versus python

data <- read.csv("py_v_ib.csv")
data |> head()
lapply(data |> select(-c("wall_s")), unique)

data |>
  group_by(network, algorithm, query_size, framework) |>
  summarize(n = n())


data |>
  ggplot(aes(x = framework, y = wall_s)) +
  facet_grid(
    rows = vars(algorithm),
    cols = vars(network),
    scales = "free_y"
  ) +
  geom_col(data = ~ filter(.x, algorithm == "shell_offline")) +
  geom_boxplot(data = ~ filter(.x, algorithm != "shell_offline")) +
  theme_bw() +
  theme(
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14)
  )
ggsave("py_v_ib.pdf", dpi = 300)

# ================= COMMUNITY SEARCH TESTING ===============

cs_online <- read.csv("testing-commsearch.csv")
cs_offline <- read.csv("testing-offline.csv")

cs_online |> head()
cs_online |>
  filter(stat == "exit_code", value != 0)
lapply(cs_online |> select(-c("value")), table)

cs_offline |> head()
cs_offline |>
  filter(stat == "exit_code", value != 0)
lapply(cs_offline |> select(-c("value")), table)

## identify missing data in cs_online:

# csk is missing all for friendster and twitter_social (out-of-memory)
cs_online |>
  filter(method == "csk", stat == "exit_code", value == 0) |>
  group_by(network, method, n, b) |>
  summarize(count = n())

# we expect 7 (networks) x 2 (methods) x 4 (n) x 3 (b) = 168 rows
# no missing data for steiner and shellstruct
cs_online |>
  filter(method != "csk", stat == "exit_code", value == 0) |>
  group_by(network, method, n, b) |>
  summarize(count = n())

data <- cs_online |>
  filter(stat %in% c("exit_code", "wall_s", "peak_rss_kb")) |>
  pivot_wider(names_from = stat, values_from = value) |>
  left_join(
    cs_offline |>
      filter(stat %in% c("exit_code", "wall_s", "peak_rss_kb")) |>
      pivot_wider(names_from = stat, values_from = value) |>
      mutate(wall_s = as.numeric(wall_s), peak_rss_kb = as.numeric(peak_rss_kb)) |>
      pivot_wider(names_from = method, values_from = c(wall_s, peak_rss_kb, exit_code)),
    by = "network"
  ) |>
  filter(exit_code == 0) |>
  mutate(
    offline_wall_s = case_when(
      method %in% c("csk", "steiner") ~ `wall_s_ib-core-decomp`,
      method == "par-shellstruct" ~ `wall_s_ib-core-decomp` + `wall_s_par-shellstruct-offline`
    ),
    offline_peak_rss_kb = case_when(
      method %in% c("csk", "steiner") ~ `peak_rss_kb_ib-core-decomp`,
      method == "par-shellstruct" ~ pmax(`peak_rss_kb_ib-core-decomp`, `peak_rss_kb_par-shellstruct-offline`)
    ),
    wall_s = wall_s + offline_wall_s,
    peak_rss_kb = pmax(peak_rss_kb, offline_peak_rss_kb, na.rm = TRUE)
  ) |>
  select(-starts_with("wall_s_"), -starts_with("peak_rss_kb_"), -offline_wall_s, -offline_peak_rss_kb)

## problem one:
# data |>
#   filter(n == 1, b == 1) |>
#   mutate(`Time (s)` = wall_s, `Memory (GB)` = peak_rss_kb / 1e6) |>
#   select(network, method, n, b, r, `Time (s)`, `Memory (GB)`) |>
#   pivot_longer(cols = c(`Time (s)`, `Memory (GB)`)) |>
#   ggplot(aes(x = network, y = value, fill = method)) +
#   geom_boxplot() +
#   facet_grid(
#     rows = vars(name),
#     scales = "free_y",
#   ) +
#   scale_y_continuous(transform = "log10", limits = c(1, NA)) +
#   theme_bw()

FAILURES <- bind_rows(
  expand_grid(
    network = c("friendster", "twitter_social", "wikipedia_link"),
    method = "shellstruct",
    n = c(1, 5, 10, 20),
    b = c(1, 10, 100)
  ) |>
    mutate(reason = "oom"),
  expand_grid(
    network = c("bitcoin", "dbpedia_link", "livejournal", "microsoft_concept"),
    method = "shellstruct",
    n = c(1, 5, 10, 20),
    b = c(1, 10, 100)
  ) |>
    mutate(reason = "timeout"),
  expand_grid(
    network = c("friendster", "twitter_social"),
    method = "csk",
    n = c(1),
    b = c(1, 10, 100)
  ) |>
    mutate(reason = "oom"),
)

METHODS <- c("steiner", "par-shellstruct", "csk", "shellstruct")
NETWORKS <- c("livejournal", "bitcoin", "wikipedia_link", "microsoft_concept", "dbpedia_link", "twitter_social", "friendster")

plot_data <- data |>
  mutate(`Time (s)` = wall_s, `Memory (GB)` = peak_rss_kb / 1e6) |>
  pivot_longer(cols = c(`Time (s)`, `Memory (GB)`)) |>
  group_by(network, method, name, n, b) |>
  summarize(mean = mean(value, na.rm = TRUE), se = sd(value, na.rm = TRUE) / sqrt(sum(!is.na(value))), .groups = "drop") |>
  full_join(
    FAILURES |> expand_grid(name = c("Time (s)", "Memory (GB)")),
    by = c("network", "method", "name", "n", "b")
  ) |>
  mutate(
    is_missing = !is.na(reason),
    method = factor(method, levels = METHODS),
    network = factor(network, levels = NETWORKS)
  ) |>
  group_by(name) |>
  mutate(ghost = max(mean, na.rm = TRUE)) |>
  ungroup()

do_plot <- function(df) {
  df |>
    ggplot(aes(x = network, fill = method)) +
    geom_col(
      aes(y = if_else(is_missing, ghost, mean), alpha = is_missing),
      position = position_dodge2(width = 0.9, preserve = "single"),
    ) +
    geom_text(
      aes(y = ghost, label = reason),
      position = position_dodge2(width = 0.9, preserve = "single"),
      vjust = 1.3, size = 3, angle = 45
    ) +
    scale_alpha_manual(values = c(`FALSE` = 1, `TRUE` = 0.15), guide = "none") +
    facet_grid(rows = vars(name), scales = "free_y") +
    scale_y_continuous(transform = "log10", limits = c(1, NA), name = "") +
    scale_x_discrete(name = "") +
    theme_bw() +
    theme(
      strip.text = element_text(size = 16, face = "bold"),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14, angle = 30, hjust = 1)
    )
}

do_plot(plot_data |> filter(n == 1, b == 1))
ggsave("commsearch-n1-b1.pdf")
do_plot(plot_data |> filter(n == 1, b == 10))
ggsave("commsearch-n1-b10.pdf")
do_plot(plot_data |> filter(n == 1, b == 100))
ggsave("commsearch-n1-b100.pdf")

do_plot(plot_data |> filter(n == 5, b == 1))
ggsave("commsearch-n5-b1.pdf")
do_plot(plot_data |> filter(n == 5, b == 10))
ggsave("commsearch-n5-b10.pdf")
do_plot(plot_data |> filter(n == 5, b == 100))
ggsave("commsearch-n5-b100.pdf")

do_plot(plot_data |> filter(n == 10, b == 1))
ggsave("commsearch-n10-b1.pdf")
do_plot(plot_data |> filter(n == 10, b == 10))
ggsave("commsearch-n10-b10.pdf")
do_plot(plot_data |> filter(n == 10, b == 100))
ggsave("commsearch-n10-b100.pdf")

do_plot(plot_data |> filter(n == 20, b == 1))
ggsave("commsearch-n20-b1.pdf")
do_plot(plot_data |> filter(n == 20, b == 10))
ggsave("commsearch-n20-b10.pdf")
do_plot(plot_data |> filter(n == 20, b == 100))
ggsave("commsearch-n20-b100.pdf")

# ======================= community evolution in ABM 272

data <- read.csv("comm-evolve-stats.csv")
data |> head(n = 16)
data |> tail(n = 16)

histogram <- tribble(
  ~field, ~n,
  "math", 12028110,
  "physics", 67483337,
  "chemistry", 41087690,
  "bio", 138978896,
  "scientometrics", 13074150,
  "math|scientometrics", 42230,
  "physics|chemistry", 8046,
  "physics|scientometrics", 14047,
  "chemistry|bio", 21215,
  "bio|scientometrics", 1762
)

fields <- c("math", "physics", "chemistry", "bio", "scientometrics")
total_n <- sum(histogram$n)

fraction <- map_dfr(fields, function(f) {
  tibble(
    query = f,
    frac = sum(histogram$n[str_detect(histogram$field, f)]) / sum(histogram$n)
  )
})

data |>
  mutate(
    query = factor(
      query,
      levels = c("838", "9326", "16456", "24968", "26589"),
      labels = c("math", "bio", "chemistry", "physics", "scientometrics")
    )
  ) |>
  filter(year <= 2100) |>
  group_by(year, query) |>
  summarize(
    coreness = first(value[key == "coreness"]),
    size = sum(value[key != "coreness"]),
    math = sum(value[str_detect(key, "math")]),
    physics = sum(value[str_detect(key, "physics")]),
    chemistry = sum(value[str_detect(key, "chemistry")]),
    bio = sum(value[str_detect(key, "bio")]),
    scientometrics = sum(value[str_detect(key, "scientometrics")]),
    prop_largest = pmax(math, physics, chemistry, bio, scientometrics) / size,
    prop_same = sum(value[str_detect(key, as.character(query))]) / size,
    `log10(size)` = log10(size),
  ) |>
  select(year, query, `log10(size)`, coreness, prop_same) |>
  left_join(fraction, by = "query") |>
  mutate(prop_same_over_baseline = prop_same / frac) |>
  pivot_longer(c("log10(size)", "coreness", "prop_same_over_baseline")) |>
  mutate(name = factor(name, levels = c("log10(size)", "coreness", "prop_same_over_baseline"))) |>
  ggplot(aes(x = year, y = value, color = factor(query))) +
  facet_grid(
    row = vars(name), scales = "free_y",
    labeller = as_labeller(c(`coreness` = "min degree", `prop_same_over_baseline` = "field enrichment", `log10(size)` = "log10(size)"))
  ) +
  facetted_pos_scales(
    y = list(
      name == "log10(size)" ~ scale_y_continuous(name = NULL, limits = c(1, NA)),
      name == "coreness" ~ scale_y_continuous(name = NULL, limits = c(0, NA)),
      name == "prop_same_over_baseline" ~ scale_y_continuous(name = NULL, transform = "log10")
    )
  ) +
  scale_x_continuous(name = NULL, breaks = c(2026, 2050, 2075, 2100)) +
  geom_line() +
  theme_bw() +
  theme(
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.text = element_text(size = 16),
  )
ggsave("comm-evolve.pdf")
