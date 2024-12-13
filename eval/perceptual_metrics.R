source("/Users/pdealcan/Documents/github/doc_suomi/code/utils.R")
library(tidyr)

file = "/Users/pdealcan/Downloads/perceptual.csv"
df = fread(file)

similarity = df %>% filter(trial_type %in% c("survey-likert"))
emotion = df %>% filter(trial_type %in% c("html-multi-slider-response"))

similarity$response = gsub("{'Q0': ", "", similarity$response, perl=TRUE)
similarity$response = gsub("}", "", similarity$response, perl=TRUE)
similarity$response = as.numeric(similarity$response)

emotion$response = gsub("\\[", "", emotion$response, perl=TRUE)
emotion$response = gsub("]", "", emotion$response, perl=TRUE)

emotion$video = gsub("singles/control/true_", "", emotion$video, perl=TRUE)
emotion$video = gsub("singles/experiment/pred_", "", emotion$video, perl=TRUE)

emotion = emotion %>%
  mutate(response = strsplit(response, ",\\s*")) %>%
  # Unnest the list column into separate columns
  unnest_wider(response, names_sep = "") %>%
  # Rename the columns
  rename(valence = response1, arousal = response2, quality = response3) %>%
  select(valence, arousal, quality, trial_index, condition, video, startDateJATOS, duration)


#Visualizations
similarity %>%
  ggplot(aes(x=condition, y=response, fill = condition))+
    geom_boxplot()

similarity %>%
  ggplot(aes(x=response, fill = condition))+
    geom_density(alpha = 0.5) +
    xlab("Similarity")

#ggsave("/Users/pdealcan/Documents/github/data/CoE/accel/perceptual_experiment/similarity.png")
emotion = emotion %>%
  arrange(video) %>%
  select(condition, video, valence, arousal, quality, startDateJATOS) %>%
  melt(id.vars = c("condition", "video", "startDateJATOS")) %>%
  mutate(value = as.numeric(value)) %>%
  group_by(condition, startDateJATOS, variable) %>%
  summarize(diff = abs(diff(value)), .groups = 'drop')

a = emotion %>%
  group_by(condition, variable) %>%
  summarise(mean_diff = mean(diff, na.rm = TRUE), 
            stder = sd(diff, na.rm = TRUE) / sqrt(n()))

emotion %>%
  ggplot() +
    geom_jitter(aes(x = condition, y = diff, color = condition), width = 0.2, alpha = 0.5) +
    geom_point(data = a, aes(x = condition, y = mean_diff), size = 3, color = "black") +
    geom_errorbar(data = a, aes(x = condition, ymin = mean_diff - stder, ymax = mean_diff + stder), width = 0.2, color = "black") +
    facet_wrap(~variable)+
    labs(title = "",
         x = "",
         y = "Emotion difference")

#ggsave("/Users/pdealcan/Documents/github/data/CoE/accel/perceptual_experiment/emotion_difference.png")
