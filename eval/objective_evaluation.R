source("/Users/pdealcan/Documents/github/doc_suomi/code/utils.R")

process = function(directory){
  final = list()

  file_list <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  # Read and combine all files into one data frame
  df <- file_list %>%
    lapply(read.csv) %>%# Read each file as a data frame
    bind_rows()# Combine all data frames by rows

  mpe = df %>%
    select("condition", "experiment_run", "root", "rhip", "lhip", "belly", "rknee", "lknee", "lchest","rankle", "lankle","upchest", "rtoe",  "ltoe", "neck",  "rclavicle", "lclavicle", "head", "rshoulder", "lshoulder", "relbow","lelbow", "rwrist","lwrist", "rhand", "lhand") #%>%
#    group_by(condition, experiment_run) %>%
#    summarise(across(everything(), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
#    rowwise() %>%
#    mutate(mean = mean(c(root, rhip, lhip, belly, rknee, lknee, lchest, rankle, lankle, upchest, rtoe,  ltoe, neck,  rclavicle, lclavicle, head, rshoulder, lshoulder, relbow, lelbow, rwrist, lwrist, rhand, lhand))) %>%
#    select(condition, experiment_run, mean) %>%
#    arrange(condition, experiment_run)

  gtc = df %>%
    select("condition", "experiment_run", "gtc_first", "gtc_second", "gtc_third") #%>%
#    group_by(condition, experiment_run) %>%
#    summarise(across(everything(), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
#    rowwise()# %>%
#    mutate(mean = mean(c(gtc_first, gtc_second, gtc_third))) %>%
#    select(condition, experiment_run, mean)
  
  final$mpe = mpe
  final$gtc = gtc
  return(final)
}

directory = "./eval_data/objective_eval_exp1/"
exp1 = process(directory)

directory = "./eval_data/objective_eval_exp2/"
exp2 = process(directory)

directory = "./eval_data/objective_eval_exp3/"
exp3 = process(directory)

directory = "./eval_data/objective_eval_exp4/"
exp4 = process(directory)

directory = "./eval_data/objective_eval_exp5/"
exp5 = process(directory)

mpe = bind_rows(exp1$mpe, exp2$mpe, exp3$mpe, exp4$mpe, exp5$mpe)
mpe = bind_rows(exp3$mpe, exp5$mpe)
mpe = mpe %>% 
  separate(
    col = experiment_run,      # The column to split
    into = c("experiment", "epoch"), # New column names
    sep = "_epoch_",      # Separator string
  ) %>%
  mutate(across(c(experiment, epoch), ~ replace_na(., "7600"))) # Replace NA with "1"

gtc = bind_rows(exp1$gtc, exp2$gtc, exp3$gtc, exp4$gtc, exp5$gtc)
gtc = bind_rows(exp3$gtc, exp5$gtc)
gtc = gtc %>%
  separate(
    col = experiment_run,      # The column to split
    into = c("experiment", "epoch"), # New column names
    sep = "_epoch_",      # Separator string
  ) %>%
  mutate(across(c(experiment, epoch), ~ replace_na(., "7600"))) # Replace NA with "1"

gtc %>%
  melt(id.vars = c("condition", "experiment", "epoch")) %>%
  group_by(experiment) %>%
  filter(epoch == max(epoch)) %>%
  ggplot(aes(x = 1, y = value, color = condition)) +
    facet_wrap(~experiment) +
    geom_boxplot()

gtc %>%
  melt(id.vars = c("condition", "experiment", "epoch")) %>%
  group_by(experiment) %>%
  filter(epoch == max(epoch)) %>%
  ggplot(aes(x = value, fill = condition)) +
    geom_density(alpha = 0.6) +
    facet_wrap(~experiment) +
    theme_minimal() +
    labs(
      title = "Density Plot of Values by Condition and Experiment",
      x = "Value",
      y = "Density"
    )

mpe %>%
  melt(id.vars = c("condition", "experiment", "epoch")) %>%
  group_by(experiment) %>%
  filter(epoch == max(epoch)) %>%
  ggplot(aes(x = experiment, y = value, color = condition)) +
    geom_boxplot()

gtc %>%
  melt(id.vars = c("condition", "experiment", "epoch")) %>%
  group_by(experiment) %>%
  filter(epoch == max(epoch)) %>%
  ggplot(aes(x = variable, y = value, fill = condition)) +
    facet_wrap(~experiment) +
    geom_violin(alpha = 0.8, scale = "width", trim = TRUE) +
    geom_boxplot(width = 0.2, position = position_dodge(0.9), color = "black", outlier.shape = NA) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "top") +
    scale_fill_brewer(palette = "Set2") +  # Use a color palette for discrete values
    labs(
      title = "Violin Plot by Condition and Experiment",
      x = "Variable",
      y = "Value"
    )
#exp 3 and 5 are the best

#exp2 melhor em MPE
#exp5 melhor em GTC


# Step 1: Calculate the mean for control and experimental conditions for each variable and experiment
gtc_means <- gtc %>%
  melt(id.vars = c("condition", "experiment", "epoch")) %>%
  filter(epoch == max(epoch)) %>%
  group_by(variable, experiment, condition) %>%
  summarise(mean_value = mean(value), .groups = "drop")

# Step 2: Separate the data for control and experimental conditions
control_data <- gtc_means %>%
  filter(condition == "control")

experimental_data <- gtc_means %>%
  filter(condition == "experiment")

# Step 3: Merge the control and experimental data on variable and experiment
gtc_diff <- control_data %>%
  left_join(experimental_data, by = c("variable", "experiment"), suffix = c("_control", "_experimental"))

# Step 4: Calculate the difference between the experimental and control means
gtc_diff <- gtc_diff %>%
  mutate(difference = mean_value_experimental - mean_value_control)

# Step 5: Merge the difference data back into the original data
gtc_plot <- gtc %>%
  melt(id.vars = c("condition", "experiment", "epoch")) %>%
  filter(epoch == max(epoch)) %>%
  left_join(gtc_diff, by = c("variable", "experiment"))

# Step 6: Plot with color gradient based on the difference
gtc_plot %>%
  ggplot(aes(x = variable, y = value, fill = difference)) +
    facet_wrap(~experiment) +
    geom_violin(alpha = 0.8, scale = "width", trim = TRUE) +
    geom_boxplot(width = 0.2, position = position_dodge(0.9), color = "black", outlier.shape = NA) +
    scale_fill_gradient2(low = "white", mid = "#56B1F7", high = "#FF6F61") +
    theme_minimal(base_size = 14) +
    labs(
      title = "Violin Plot Colored by Difference (Experimental - Control)",
      x = "Variable",
      y = "Value"
    )



