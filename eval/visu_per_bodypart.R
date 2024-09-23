source("/Users/pdealcan/Documents/github/doc_suomi/code/utils.R")

df = fread("./eval_data/mpe_statistics_final.csv")

df %>%
  group_by(condition, face_name) %>%
  summarise(m = mean(m), sd = mean(sd)) %>%
  write.csv("./processed_mpe.csv")

