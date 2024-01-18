library(dplyr)
library(data.table)
library(ggplot2)

accel = fread("./Accelerometer.csv")
wrist = fread("./WristMotion.csv")

accel = accel %>% select(time, x, y, z)
wrist = wrist %>% select(time, accelerationX, accelerationY, accelerationZ)

colnames(wrist) = c("time", "x", "y", "z")

accel$sensor = "phone"
wrist$sensor = "watch"

accel = accel %>% filter(time >= min(wrist$time))
#8774
length(accel$x)
df = bind_rows(accel, wrist)

plot(1:length(df$time), df$time)

df %>%
  ggplot(aes(x = as.integer(time), y = y, color = sensor))+
    facet_wrap(~sensor)+
    geom_point()


min(wrist$time) < max(accel$time)
