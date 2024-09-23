source("/Users/pdealcan/Documents/github/doc_suomi/code/utils.R")
library(tidyr)

file = "/Users/pdealcan/Documents/github/data/CoE/accel/perceptual_experiment/perceptual.csv"
df = fread(file)

df = df %>% filter(trial_type == "survey-likert")

df$response = gsub("{'Q0': ", "", df$response, perl=TRUE)
df$response = gsub("}", "", df$response, perl=TRUE)
df$response = as.numeric(df$response)

df %>%
  ggplot(aes(x=condition, y=response, fill = condition))+
    geom_boxplot()

df %>%
  ggplot(aes(x=response, fill = condition))+
    geom_density(alpha = 0.5) +
    xlab("Similarity")

ggsave("/Users/pdealcan/Documents/github/data/CoE/accel/perceptual_experiment/similarity.png")

#Calculate emotion responses
df = fread(file)
df = df %>% filter(trial_type == "html-multi-slider-response")

df$video = gsub("singles/control/true_", "", df$video, perl=TRUE)
df$video = gsub("singles/experiment/pred_", "", df$video, perl=TRUE)
df$response = gsub("\\[", "", df$response, perl=TRUE)
df$response = gsub("]", "", df$response, perl=TRUE)
df$response = gsub(" ", "", df$response, perl=TRUE)

df = df %>%
  arrange(video) %>%
  separate(response, into = c("valence", "arousal"), sep = ",") %>%
  select(condition, video, valence, arousal, startDateJATOS) %>%
  melt(id.vars = c("condition", "video", "startDateJATOS")) %>%
  mutate(value = as.numeric(value)) %>%
  group_by(condition, startDateJATOS, variable) %>%
  summarize(diff = abs(diff(value)), .groups = 'drop')

a = df %>%
  group_by(condition, variable) %>%
  summarise(mean_diff = mean(diff), 
            stder = sd(diff)/sqrt(length(diff)))

df %>%
  ggplot(aes(x = condition, y = diff, fill = condition)) +
    facet_wrap(~variable)+
    geom_jitter()+
    geom_point(aes(x = condition, y = mean_diff), data = a)+
    geom_errorbar(aes(ymin = mean_diff-stder, ymax = mean_diff+stder), data = a)

df %>%
  ggplot() +
    geom_jitter(aes(x = condition, y = diff, color = condition), width = 0.2, alpha = 0.5) +
    geom_point(data = a, aes(x = condition, y = mean_diff), size = 3, color = "black") +
    geom_errorbar(data = a, aes(x = condition, ymin = mean_diff - stder, ymax = mean_diff + stder), width = 0.2, color = "black") +
    facet_wrap(~variable)+
    labs(title = "",
         x = "",
         y = "Emotion difference")

ggsave("/Users/pdealcan/Documents/github/data/CoE/accel/perceptual_experiment/emotion_difference.png")
