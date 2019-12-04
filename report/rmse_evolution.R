library(ggplot2)
library(dplyr)
library(plotly)

init.date <- as.Date("2019-10-25")
init.x <- init.date + c(0, 1, 15, 18, 20)
init.y <- c(0.85, 0.84, 0.83, 0.65, 0.39)

init.df <- data.frame(
  x = init.x,
  y = init.y
)

divergence.date <- as.Date("2019-11-14")

continue.x <- divergence.date + c(5, 7, 10, 19)
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
  'Distance between airports', 
  'GradientBoosting',
  'Hyper-parameter Tuning',
  'HistGradientBoosting',
  'AdaBoost'
)
normal.notables[, 'ax'] <- c(
  30,
  60,
  40,
  40,
  60,
  -40
)
normal.notables[, 'ay'] <- c(
  30,
  -40,
  -60,
  -40,
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

cheat.x <- divergence.date + c(5, 6, 12)
cheat.y <- c(0.39, 0.14, 0)
cheat.df <- data.frame(
  x = cheat.x,
  y = cheat.y
)
cheat.df[, 'is_notable'] <- c(F, T, F)
cheat.notables <- cheat.df %>% filter(is_notable)
cheat.notables[, 'annotation'] <- c('Training on test data + noise')
cheat.notables[, 'ax'] <- c(-40)
cheat.notables[, 'ay'] <- c(30)

cheat.annotation <- list(
  x = cheat.notables$x,
  y = cheat.notables$y,
  text = cheat.notables$annotation,
  xref = "x",
  yref = "y",
  showarrow = TRUE,
  arrowhead = 7,
  arrowsize = .5,
  ax = cheat.notables$ax,
  ay = cheat.notables$ay
)

plot_ly(
  normal.df, 
  x = ~x, 
  y = ~y, 
  type = "scatter", 
  mode = 'lines+markers', 
  name = 'Normal models'
) %>% 
  layout(
    annotations = normal.annotation,
    xaxis = list(
      title = "Time",
      titlefont = F
    ),
    yaxis = list(
      title = "RMSE",
      titlefont = F
    )
  ) %>% 
  add_trace(
    x = cheat.df$x, 
    y = cheat.df$y, 
    col, 
    mode = 'lines+markers',
    name = 'Cheat'
  ) %>% 
  layout(
    annotations = cheat.annotation
  )
