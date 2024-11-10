source("/Users/pdealcan/Documents/github/doc_suomi/code/utils.R")

directory = "./eval_data/objective_eval_exp3/"
file_list <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)

# Read and combine all files into one data frame
df <- file_list %>%
  lapply(read.csv) %>%# Read each file as a data frame
  bind_rows()# Combine all data frames by rows

df %>%
  select("condition", "experiment_run", "root", "rhip", "lhip", "belly", "rknee", "lknee", "lchest","rankle", "lankle","upchest", "rtoe",  "ltoe", "neck",  "rclavicle", "lclavicle", "head", "rshoulder", "lshoulder", "relbow","lelbow", "rwrist","lwrist", "rhand", "lhand") %>%
  group_by(condition, experiment_run) %>%
  summarise(across(everything(), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  rowwise() %>%
  mutate(mean = mean(c(root, rhip, lhip, belly, rknee, lknee, lchest, rankle, lankle, upchest, rtoe,  ltoe, neck,  rclavicle, lclavicle, head, rshoulder, lshoulder, relbow, lelbow, rwrist, lwrist, rhand, lhand))) %>%
  select(condition, experiment_run, mean)

df %>%
  select("condition", "experiment_run", "gtc_first", "gtc_second", "gtc_third") %>%
  group_by(condition, experiment_run) %>%
  summarise(across(everything(), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  rowwise() %>%
  mutate(mean = mean(c(gtc_first, gtc_second, gtc_third))) %>%
  select(condition, experiment_run, mean)

#exp4_epoch_1000 1.26 
#exp4_epoch_4100 0.361
#exp4_epoch_7600 0.283

#exp3_epoch_1000 0.359
#exp3_epoch_4100 0.225
#exp3_epoch_7600 0.226
