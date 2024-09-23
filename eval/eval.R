source("/Users/pdealcan/Documents/github/doc_suomi/code/utils.R")
library("xtable")
library("afex")

control = fread("./eval_data/objective_measures/objective_measure_control.csv")
experimental = fread("./eval_data/objective_measures/objective_measure_experimental.csv")

df = bind_rows(control, experimental)

nNames = colnames(df)
nNames[25] = "x"
nNames[26] = "y"
nNames[27] = "z"

colnames(df) = nNames
#GTC 
df %>% 
  select(condition, x, y, z) %>%
  melt() %>%
  ggplot(aes(x = variable, y = value, fill = condition)) +
    geom_boxplot() +
    labs(x = NULL, y = "GTC", fill = "Condition") +
    theme_minimal()+
    theme(legend.position = "bottom")+
    labs(fill = NULL)   

ggsave("./gtc.png")

gtc = fread("/Users/pdealcan/Documents/github/edGEk/eval/eval_data/objective_measures/gtc_full.csv")
unique(gtc$file)
gtc %>% 
  melt(id.vars = c("file")) %>%
  separate(variable, c('dimension', 'condition')) %>%
  group_by(condition, file, dimension) %>%
  mutate(index = seq(1, length(value))) %>%
  filter(file == 677) %>%
  ggplot(aes(x = index, y = value, color = condition))+
    facet_wrap(~dimension)+
    geom_path()+
    xlab("Frame")+
    ylab("Position")+
    theme(legend.position = "bottom")

ggsave("./b.png", width = 4, height = 3)


#GTC statistics
gtc_anova = df %>%
  select(file, condition, x, y, z) %>%
  melt(id.vars = c("file", "condition")) %>%
  group_by(file, condition, variable) %>%
  summarise(value = mean(value))

res.aov <- aov(value ~ condition*variable, data = gtc_anova)
summary(res.aov)

TukeyHSD(res.aov)
print()






#Mean error
df %>%
  select(condition, file, root, rhip, lhip, belly, rknee, lknee, lchest, rankle, lankle, upchest, rtoe, ltoe, neck, rclavicle, lclavicle, head, rshoulder, lshoulder, relbow, lelbow, rwrist, lwrist, rhand, lhand) %>%
  melt() %>%
  ggplot(aes(x = variable, y = value, fill = condition)) +
    geom_boxplot() +
    labs(x = NULL, y = "MPE", fill = "Condition") +
    theme_minimal()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    theme(legend.position = "bottom")+
    labs(fill = NULL)   

df %>%
  select(condition, file, root, rhip, lhip, belly, rknee, lknee, lchest, rankle, lankle, upchest, rtoe, ltoe, neck, rclavicle, lclavicle, head, rshoulder, lshoulder, relbow, lelbow, rwrist, lwrist, rhand, lhand) %>%
  melt() %>%
  ggplot(aes(x = variable, y = value, fill = condition)) +
    geom_boxplot() +
    labs(x = NULL, y = "MPE", fill = "Condition") +
    theme_minimal()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    theme(legend.position = "bottom")+
    labs(fill = NULL)   

ggsave("./mpe.png")

#Descriptives of mpe
df = df %>%
  select(condition, file, root, rhip, lhip, belly, rknee, lknee, lchest, rankle, lankle, upchest, rtoe, ltoe, neck, rclavicle, lclavicle, head, rshoulder, lshoulder, relbow, lelbow, rwrist, lwrist, rhand, lhand) %>%
  melt()

joint2face = fread("./joint_face_correspondance.csv")
df = merge(df, joint2face, by.x = "variable", by.y = "variable")

experiment = df %>%
  group_by(variable, condition, face_name) %>%
  summarise(m = mean(value), sd = sd(value)) %>%
  filter(condition == "experiment")

control = df %>%
  group_by(variable, condition, face_name) %>%
  summarise(m = mean(value), sd = sd(value)) %>%
  filter(condition == "control")

diff = control
diff$m = control$m - experiment$m
diff$sd = control$sd - experiment$sd
write.csv(experiment, "./mpe_statistics_experiment.csv")
write.csv(control, "./mpe_statistics_control.csv")
write.csv(diff, "./mpe_statistics_diff.csv")

df %>% 
  select(condition, file, root, rhip, lhip, belly, rknee, lknee, lchest, rankle, lankle, upchest, rtoe, ltoe, neck, rclavicle, lclavicle, head, rshoulder, lshoulder, relbow, lelbow, rwrist, lwrist, rhand, lhand) %>%
  melt() %>%
  group_by(condition, variable) %>%
  summarise(mean = mean(value), stder = sd(value)) %>%
  mutate(value = paste(round(mean, 2), "(", round(stder, 2), ")", sep = "")) %>%
  select(condition, variable, value) %>%
  pivot_wider(names_from = condition, values_from = value) %>%
  xtable()

#Personality
a = df %>% select(file, condition, bound_rect_diff, cm_acc_diff, cm_jerk_diff, acc_diff_feet_head_hand, jerk_diff_feet_head_hand, v_diff_torso_feet_head_hand)
colnames(a) = c("file", "condition", "Bount Rectangle", "CoM Acc", "CoM Jerk", "Acc Feet Head Hand", "Jerk Feet Head Hand", "Velocity Torso Feet Head Hand")
#select(file, condition, bound_rect_diff, cm_acc_diff, cm_jerk_diff, acc_diff_feet_head_hand, jerk_diff_feet_head_hand, v_diff_torso_feet_head_hand) %>%
a %>% 
  select(file, condition, `Bount Rectangle`, `CoM Acc`, `CoM Jerk`, `Acc Feet Head Hand`, `Jerk Feet Head Hand`, `Velocity Torso Feet Head Hand`) %>%
  melt() %>%
  ggplot(aes(x = condition, y = value, fill = condition)) +
    geom_boxplot() +
    facet_wrap(~variable, scale = "free")+
    labs(x = NULL, y = "Difference scores", fill = "Condition") +
    theme_minimal()+
    theme(legend.position = "none")+
    labs(fill = NULL)   

ggsave("./personality.png")
