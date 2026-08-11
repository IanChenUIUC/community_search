library(dplyr)
library(tidyverse)
library(ggh4x)
library(xtable)
library(ggpattern)

## runtime of our methods and SoTA
## two subfigures, for single query and multiple query
## runtime on dbpedia_link (web graph, based on wikipedia)

cs_online <- read.csv("testing-commsearch.csv") |>
  group_by(network, method, n, b) |>
  filter(all(value[stat == "exit_code"] == 0)) |>
  ungroup() |>
  filter(stat == "wall_s", network == "dbpedia_link") |>
  pivot_wider(names_from = stat, values_from = value)
cs_offline <- read.csv("testing-offline.csv") |>
  group_by(network, method) |>
  filter(all(value[stat == "exit_code"] == 0)) |>
  ungroup() |>
  filter(stat == "wall_s", network == "dbpedia_link") |>
  pivot_wider(names_from = stat, values_from = value)

cs_online |> head()
cs_offline |> head()

METHODS <- c("steiner", "par-shellstruct", "csk", "shellstruct")

data <- cs_online |>
  mutate(
    group = factor(if_else(n > 1, "multi", "single"),
      levels = c("single", "multi")
    )
  ) |>
  group_by(network, method, group, b) |>
  summarize(
    wall_s_q1 = quantile(wall_s, .25),
    wall_s_q3 = quantile(wall_s, .75),
    wall_s = mean(wall_s),
    .groups = "drop"
  ) |>
  left_join(
    cs_offline |> pivot_wider(names_from = method, values_from = wall_s),
    by = "network"
  ) |>
  mutate(
    offline_add = case_when(
      method %in% c("csk", "steiner") ~ `ib-core-decomp`,
      method == "par-shellstruct" ~ `ib-core-decomp` + `par-shellstruct-offline`
    ),
    wall_s = wall_s + offline_add,
    wall_s_q1 = wall_s_q1 + offline_add,
    wall_s_q3 = wall_s_q3 + offline_add
  ) |>
  select(network, method, group, b, wall_s, wall_s_q1, wall_s_q3) |>
  mutate(
    method = factor(method, levels = METHODS),
    group = factor(group, levels = c("single", "multi")),
  )

data |> head()
data |> tail()

data |>
  filter(group == "single") |>
  ggplot(aes(x = factor(b), y = wall_s, fill = method)) +
  facet_grid(
    rows = vars(group),
    labeller = labeller(group = c(`single` = "single-vertex query", `multi` = "multiple-vertex query"))
  ) +
  geom_col_pattern(
    aes(pattern = is_missing, alpha = is_missing),
    colour = "grey20",
    pattern_fill = "grey50",
    pattern_colour = "grey50",
    pattern_angle = 45,
    pattern_density = 0.05,
    pattern_spacing = 0.02,
    pattern_key_scale_factor = 0.5,
    position = position_dodge2(width = 0.9, preserve = "single")
  ) +
  scale_pattern_manual(values = c(`FALSE` = "none", `TRUE` = "crosshatch"), guide = "none") +
  geom_text(
    aes(label = label),
    position = position_dodge2(width = 0.9, preserve = "single"),
    colour = "red",
    vjust = -0.2
  ) +
  scale_pattern_manual(values = c(`FALSE` = "none", `TRUE` = "crosshatch"), guide = "none") +
  scale_alpha_manual(values = c(`FALSE` = 1, `TRUE` = 0.75), guide = "none") +
  scale_y_continuous(transform = "log10", limits = c(1, NA), name = "Time (s)") +
  scale_x_discrete(name = "number of queries") +
  theme_bw() +
  theme(
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    axis.text.x = element_text(size = 14),
    legend.text = element_text(size = 16),
    legend.title = element_blank(),
    legend.position = "bottom",
  ) +
  guides(fill = guide_legend(override.aes = list(pattern = "none", alpha = 1)))
ggsave("compare-single.pdf", width = 10, height = 4)

data |>
  filter(group == "multi") |>
  ggplot(aes(x = factor(b), y = wall_s, fill = method)) +
  facet_grid(
    rows = vars(group),
    labeller = labeller(group = c(`single` = "single-vertex query", `multi` = "multiple-vertex query"))
  ) +
  geom_col_pattern(
    aes(pattern = is_missing, alpha = is_missing),
    colour = "grey20",
    pattern_fill = "grey50",
    pattern_colour = "grey50",
    pattern_angle = 45,
    pattern_density = 0.05,
    pattern_spacing = 0.02,
    pattern_key_scale_factor = 0.5,
    position = position_dodge2(width = 0.9, preserve = "single")
  ) +
  scale_pattern_manual(values = c(`FALSE` = "none", `TRUE` = "crosshatch"), guide = "none") +
  geom_text(
    aes(label = label),
    position = position_dodge2(width = 0.9, preserve = "single"),
    colour = "red",
    vjust = -0.2
  ) +
  scale_pattern_manual(values = c(`FALSE` = "none", `TRUE` = "crosshatch"), guide = "none") +
  scale_alpha_manual(values = c(`FALSE` = 1, `TRUE` = 0.25), guide = "none") +
  scale_y_continuous(transform = "log10", limits = c(1, NA), name = "Time (s)") +
  scale_x_discrete(name = "number of queries") +
  theme_bw() +
  theme(
    strip.text = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    axis.text.x = element_text(size = 14),
    legend.text = element_text(size = 16),
    legend.title = element_blank(),
    legend.position = "bottom",
  ) +
  guides(fill = guide_legend(override.aes = list(pattern = "none", alpha = 1)))
ggsave("compare-multiple.pdf", width = 10, height = 4)

# data |>
#   ggplot(aes(x = factor(b), y = wall_s, fill = method)) +
#   facet_grid(
#     rows = vars(group),
#     labeller = labeller(group = c(`single` = "single-vertex query", `multi` = "multiple-vertex query"))
#   ) +
#   geom_col_pattern(
#     aes(pattern = is_missing, alpha = is_missing),
#     colour = "grey20",
#     pattern_fill = "grey50",
#     pattern_colour = "grey50",
#     pattern_angle = 45,
#     pattern_density = 0.05,
#     pattern_spacing = 0.02,
#     pattern_key_scale_factor = 0.5,
#     position = position_dodge2(width = 0.9, preserve = "single")
#   ) +
#   scale_pattern_manual(values = c(`FALSE` = "none", `TRUE` = "crosshatch"), guide = "none") +
#   geom_text(
#     aes(label = label),
#     position = position_dodge2(width = 0.9, preserve = "single"),
#     colour = "red",
#     vjust = -0.05,
#   ) +
#   scale_pattern_manual(values = c(`FALSE` = "none", `TRUE` = "crosshatch"), guide = "none") +
#   scale_alpha_manual(values = c(`FALSE` = 1, `TRUE` = 0.25), guide = "none") +
#   scale_y_continuous(transform = "log10", limits = c(1, NA), name = "Time (s)") +
#   scale_x_discrete(name = "number of queries") +
#   theme_bw() +
#   theme(
#     strip.text = element_text(size = 16, face = "bold"),
#     axis.title = element_text(size = 16),
#     axis.text = element_text(size = 14),
#     axis.text.x = element_text(size = 14),
#     legend.text = element_text(size = 16),
#     legend.title = element_blank(),
#     legend.position = "bottom",
#   ) +
#   guides(fill = guide_legend(override.aes = list(pattern = "none", alpha = 1)))

data |>
  filter(method != "shellstruct") |>
  ggplot(aes(x = factor(b), y = wall_s, fill = method)) +
  facet_grid(
    rows = vars(group),
    labeller = labeller(group = c(`single` = "single-vertex query", `multi` = "multi-vertex query"))
  ) +
  geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  geom_errorbar(aes(ymin = wall_s_q1, ymax = wall_s_q3),
    position = position_dodge2(width = 0.9, preserve = "single")
  ) +
  scale_y_continuous(transform = "log10", limits = c(1, NA), name = "Time (s)") +
  scale_x_discrete(name = "number of queries") +
  scale_fill_manual(
    values = scales::hue_pal()(3),
    labels = c("steiner" = "SteinerKCore", "par-shellstruct" = "Par-Shellstruct", "csk" = "CSK")
  ) +
  theme_bw() +
  theme(
    strip.text = element_text(size = 13),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    axis.text.x = element_text(size = 14),
    legend.text = element_text(size = 16),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.box.spacing = unit(0, "pt")
  )
ggsave("compare-single-multi.pdf", width = 6, height = 5)

# ===============================

data <- read.csv("train-core-decomp.csv")
data |> head()

data |>
  filter(stat == "exit_code", value != 0)
data |>
  filter(stat == "wall_s")

lapply(data |> select(-c("value")), table)

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
ggsave("compare-core-decomp.pdf", width = 6, height = 2)
