library(dplyr)
library(tidyverse)
library(ggh4x)

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
  pivot_wider(names_from = "centrality", values_from = "cores")

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
  pivot_wider(names_from = "centrality", values_from = "cores")

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
    read.csv("abm14.analyze_shell.csv") |> mutate(network = "abm14"),
    read.csv("cen.analyze_shell.csv") |> mutate(network = "cen")
  )
head(data)
lapply(data, unique)

data |>
  ggplot(aes(x = core, y = sizes, color = network)) +
  geom_point() +
  scale_y_continuous(transform = "log10") +
  theme_bw()

## there are no "bridges", i.e. disconnected k-cores for nontrivial k
data |>
  group_by(network, core) |>
  summarize(ncomp = n()) |>
  filter(ncomp > 1)
