library(ggplot2)
library(dplyr)

init.date <- as.Date("2019-10-25")
init.x <- init.date + c(0, 1, 15, 18, 20)
init.y <- c(0.85, 0.84, 0.83, 0.75, 0.39)

init.df <- data.frame(
  x = init.x,
  y = init.y
)

divergence.date <- as.Date("2019-11-14")

continue.x <- divergence.date + c(5, 7, 10, 15)
continue.y <- c(0.39, 0.37, 0.35, 0.34)
continue.df <- data.frame(
  x = continue.x,
  y = continue.y
)

normal.df <- rbind(init.df, continue.df)
normal.df[, 'is_notable'] <- c(T, F, F, T, T, F, T, T, T)
normal.notables <- normal.df %>% filter(is_notable)
normal.notables[, 'annotation'] <- c(
  'RandomForest',
  'Added the distance between airports', 
  'GradientBoosting',
  'Hyper-parameter Tuning',
  'HistGradientBoosting',
  'AdaBoost'
)
normal.notables[, 'ax'] <- c(
  30,
  40,
  40,
  40,
  50,
  -40
)
normal.notables[, 'ay'] <- c(
  30,
  -40,
  -40,
  -30,
  -20,
  20
)

normal.annotation <- list(
  x = normal.notables$x,
  y = normal.notables$y,
  text = normal.notables$annotation,
  xref = "x",
  yref = "y",
  showarrow = TRUE,
  arrowhead = 7,
  arrowsize = .5,
  ax = normal.notables$ax,
  ay = normal.notables$ay
)
print(normal.annotation)

cheat.x <- divergence.date + c(5, 10, 15)
cheat.y <- c(0.39, 0.14, 0)
cheat.df <- data.frame(
  x = cheat.x,
  y = cheat.y
)
cheat.df[, 'is_notable'] <- c(F, F, F)


ggplot(normal.df, aes(x = x, y = y)) +
  geom_line(aes(color = "normal")) +
  geom_point(aes(color = "normal")) +
  geom_line(
    data = cheat.df, 
    aes(color = "cheat"), 
    linetype = "dashed"
  ) +
  geom_point(
    data = cheat.df, 
    aes(color = "cheat")
  ) +
  geom_text(
    x = as.Date("2019-11-22"), 
    y = 0.47,
    label = 'GradientBoostingRegressor',
    size = 4
  ) +
  geom_text(
    x = as.Date("2019-11-20"), 
    y = 0.82,
    label = 'Hyper-parameter tuning', 
    size = 4
  ) +
  geom_segment(
    data = data.frame(
      x = c(as.Date("2019-11-16"), as.Date("2019-11-14")),
      xend = c(as.Date("2019-11-14"), as.Date("2019-11-12")),
      y = c(0.44, 0.8), 
      yend = c(0.38, 0.76)
    ),
    aes(
      x = x, 
      xend = xend, 
      y = y, 
      yend = yend
    ),
    arrow = arrow(
      length = unit(0.2, "cm"), 
      type = 'closed'
    )
  ) +
  theme_light()

library(plotly)

plot_ly(
  normal.df, 
  x = ~x, 
  y = ~y, 
  type = "scatter", 
  mode = 'lines+markers', 
  name = 'Normal models'
) %>% 
  layout(
    annotations = normal.annotation
  ) %>% 
  add_trace(
    x = cheat.df$x, 
    y = cheat.df$y, 
    col, 
    mode = 'lines+markers',
    name = 'Cheat'
)
