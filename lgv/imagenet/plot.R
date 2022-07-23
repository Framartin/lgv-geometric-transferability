# install.packages('ggplot2') 
library(readr)
library(tidyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(ggforce)
library(grDevices)
library(ggpubr)
library(GGally)
library(viridis)
library(Cairo)
library(xtable)
library(kableExtra)
library(scales)

ARCHS_MAIN_PLOT <- c('ResNet-50' , 'Inception v3')
ARCHS_RESNET_FAMILY <- c("ResNet-50", "ResNet-152", "ResNext-50", "WideResNet-50")
ALL_ARCHS <- c("ResNet-50", "ResNet-152", "ResNext-50", "WideResNet-50", "DenseNet-201", "VGG19", "Inception v1", "Inception v3")


theme_set(theme_bw(base_size = 16) + theme(strip.text = element_text(size = 15)))


recode_archs <- function(chr_vec, abbr=FALSE) {
  if(!abbr) return(recode_factor(chr_vec,
                "resnet50"="ResNet-50",
                "resnet152"="ResNet-152",
                "resnext50_32x4d"="ResNext-50",
                "wide_resnet50_2"="WideResNet-50",
                "densenet201"="DenseNet-201",
                "vgg19"="VGG19",
                "googlenet"="Inception v1",
                "inception_v3"="Inception v3",
  ))
  if(abbr) return(recode_factor(chr_vec,
                          "resnet50"="RN50",
                          "resnet152"="RN152",
                          "resnext50_32x4d"="RNX50",
                          "wide_resnet50_2"="WRN50",
                          "densenet201"="DN201",
                          "vgg19"="VGG19",
                          "googlenet"="IncV1",
                          "inception_v3"="IncV3",
  ))
}

abbreviate_archs <- function(chr) {
  chr <- gsub('ResNet-50', 'RN50', chr, fixed = T)
  chr <- gsub('ResNet-152', 'RN152', chr, fixed = T)
  chr <- gsub('ResNext-50', 'RNX50', chr, fixed = T)
  chr <- gsub('WideResNet-50', 'WRN50', chr, fixed = T)
  chr <- gsub('DenseNet-201', 'DN201', chr, fixed = T)
  chr <- gsub('VGG19', 'VGG19', chr, fixed = T)
  chr <- gsub('Inception v1', 'IncV1', chr, fixed = T)
  chr <- gsub('Inception v3', 'IncV3', chr, fixed = T)
  return(chr)
}

# A function factory for getting integer y-axis values.
# source: https://www.r-bloggers.com/2019/11/setting-axes-to-integer-values-in-ggplot2/
integer_breaks <- function(n = 5, ...) {
  fxn <- function(x) {
    breaks <- floor(pretty(x, n, ...))
    names(breaks) <- attr(breaks, "labels")
    breaks
  }
  return(fxn)
}

# monkey patch to print 0 instead of 10^-Inf
get_labelling_pseudo_log <- function(label0 = '0') {
  function(x) {
    labels = math_format(expr = 10^.x, format = log10)(x)
    labels[x==0] <- label0
    return(labels)
  }
}

scientific_10 <- function(x) {
  parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))
}

#############
#  MAIN RESULTS  
#############

df <- read_csv('lgv/imagenet/attack_inter_arch.csv', show_col_types = FALSE) %>%
  mutate(
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target),
    `Target_abbr` = recode_archs(arch_target, abbr=TRUE),
    TestTimeTechs_tmp = str_replace_all(str_replace_all(transferability_techniques, '_$', ''), '_', ' + '),
    TestTimeTechs = recode(TestTimeTechs_tmp,  # old=new
     "ghost" = "GN",
     "MI + ghost" = "GN + MI",
     "MI + DI" = "DI + MI",
     "MI + SGM" = "SGM + MI",
     "DI + SGM" = "SGM + DI",
     "MI + DI + SGM" = "SGM + DI + MI",
    ),
    Surrogate_ = case_when(
      surrogate_type=='ImageNet/pretrained' ~ 'White-Box',
      !is.na(TestTimeTechs) & grepl('/original$', model_surrogate) ~ TestTimeTechs, #paste0('1 DNN + ', TestTimeTechs),
      grepl('/original$', model_surrogate) ~ '1 DNN',
      grepl('/original/noisy/std_0.005_50models', model_surrogate) ~ 'RD',
      grepl('seed[0-9]+/PCA/dims_0/noisy/std_0.01_50models', model_surrogate) ~ 'LGV-SWA + RD',
      grepl('/PCA/dims_0', model_surrogate) ~ 'LGV-SWA',
      grepl('seed[0-9]+/noisy/random_ensemble_equivalent', model_surrogate) ~ 'LGV-SWA + RD (equiv)',
      #surrogate_type=='dnn' & surrogate_size_ensembles==1 ~ '1 DNN',
      surrogate_type=='dnn' ~ paste0(surrogate_size_ensembles, ' DNNs'),
      !is.na(TestTimeTechs) & surrogate_type=='cSGD' ~ paste0('LGV + ', TestTimeTechs),
      surrogate_type=='cSGD' ~ 'LGV (ours)',
      TRUE ~ NA_character_
    )
  )
df_tmp <- df

df <- df %>%
  filter(!(surrogate_size_ensembles > 1 & grepl(' DNNs', Surrogate_))) %>%  # remove Deep Ensemble baseline 
  filter(Surrogate_ != 'White-Box') %>%
  filter(Target %in% ALL_ARCHS)

# baselines = c('White-Box', '1 DNN', paste0(2:5, ' DNNs'))
baselines = c('1 DNN')
test_time_techs = c('MI',
                    'GN', 'GN + MI',
                    'DI', 'DI + MI',
                    'SGM', 'SGM + MI', 'SGM + DI', 'SGM + DI + MI')
test_time_techs_base = grep("\\+", test_time_techs, value = TRUE, invert = TRUE)
test_time_techs_combinations = grep("\\+", test_time_techs, value = TRUE)
test_time_techs_on_lgv <- paste0('LGV + ', test_time_techs)
ours <- c('RD', 'LGV-SWA', 'LGV-SWA + RD', 'LGV (ours)')
Surrogate_levels <- c(baselines, test_time_techs, ours, test_time_techs_on_lgv)
df <- df %>%
  filter(Surrogate_ != 'LGV-SWA + RD (equiv)') %>%
  mutate(
    Surrogate = factor(Surrogate_, levels = Surrogate_levels)
  )
if(df %>% filter(is.na(Surrogate), !is.na(Surrogate_)) %>% nrow()) warning('Lost levels in conversion!')

df_ <- df %>%
  group_by(Norm, `Target_abbr`, Surrogate) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate * 100),
    sd_adv_success_rate = sd(adv_success_rate * 100),
    mean(transfer_rate* 100),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate),
  )
View(df_)
if(df_ %>% filter(n_seeds != 3) %>% nrow()) warning('Wrong number of seeds!')

norm_exported = '2'
norm_exported = 'Inf'

if (norm_exported == '2') df__ <- df_ %>% filter(Norm == 'L2') %>% ungroup()
if (norm_exported == 'Inf') df__ <- df_ %>% filter(Norm == 'L∞') %>% ungroup() 

df___ <- df__ %>%
  transmute(
    `Target_abbr`,
    Surrogate,
    success_rate_chr = paste0(formatC(mean_adv_success_rate, digits=1, format='f'), 'SMALL ±', formatC(sd_adv_success_rate, digits=1, format='f'))
  ) %>%
  pivot_wider(names_from = `Target_abbr`, values_from = success_rate_chr)
View(df___)
# xtable(df___)
#df___ %>% filter(`ResNet-50` == max(`ResNet-50`))

debug_table <- F  # if T, print HTML, otherwise output LaTeX code
n_targets <- n_distinct(df$model_target)
caption_ <- if_else(norm_exported == 2, "Success rates of baselines, state-of-the-art and LGV under the L2 attack. Simple underline is best without LGV combinations, double is best overall. Gray is LGV-based techniques worse than vanilla LGV. ``RD'' stands for random directions in the weight space. In \\%.", 
                                        "Success rates of baselines, state-of-the-art and LGV under the L∞ attack. Simple underline is best without LGV combinations, double is best overall. Gray is LGV-based techniques worse than vanilla LGV. ``RD'' stands for random directions in the weight space. In \\%.")
table <- kbl(df___, booktabs = T, caption = caption_, format = ifelse(debug_table, 'html', 'latex'), align = c('l', 'l', rep("r", n_targets)),
             table.envir = 'table*', linesep = "") %>%
  kable_styling(latex_options = c("hold_position", "striped"), font_size=9) %>% # "scale_down", "striped"
  # collapse_rows(columns = 1) %>%
  # pack_rows("Baselines", 1, length(baselines)) %>%
  pack_rows("Baselines (1 DNN)", 1, length(baselines)+length(test_time_techs)) %>%
  pack_rows("Our techniques", 1+length(baselines)+length(test_time_techs), length(baselines)+length(test_time_techs)+length(ours))  %>%
  pack_rows("LGV combined with other techniques", 1+length(baselines)+length(test_time_techs)+length(ours), nrow(df___)) %>%
  add_header_above(c(" ", "Target" = n_targets))
  # footnote(c("table footnote"))
  # str_replace(table, '\\linewidth', '\\textwidth') # patch to double columns

# add bold
for (i in 1:n_targets) {
  # best except baseline
  target_name <- colnames(df___)[1+i]
  df__best <- df__ %>% 
    # filter(Surrogate %in% c(ours, test_time_techs, test_time_techs_on_lgv), Target_abbr == target_name) %>%
    filter(Surrogate != 'White-Box', Target_abbr == target_name) %>%
    filter(mean_adv_success_rate==max(mean_adv_success_rate))
  best_cells <- df___$Surrogate %in% df__best$Surrogate
  table <- column_spec(table, 1+i, underline = best_cells) # double underline
  table <- column_spec(table, 1+i, underline = best_cells)
  df__best_alone <- df__ %>% 
    filter(Surrogate %in% c(ours, test_time_techs_base), Target_abbr == target_name) %>%
    filter(mean_adv_success_rate==max(mean_adv_success_rate))
  best_alone_cells <- df___$Surrogate %in% df__best_alone$Surrogate
  table <- column_spec(table, 1+i, underline = best_alone_cells) # simple underline
  # best_alone_colors <- rep('black', nrow(df___))
  # best_alone_colors[best_alone_cells] <- 'green'
  # table <- column_spec(table, 1+i, color = best_alone_colors)
              # color = c(rep("black", 7), "red"))
  # techs on LGV worst than LGV vanilla
  mean_adv_success_rate_lgv <- df__ %>%
    filter(Surrogate == 'LGV (ours)', Target_abbr == target_name) %>%
    select(mean_adv_success_rate)
  mean_adv_success_rate_lgv <- mean_adv_success_rate_lgv[[1]]
  df__worst_lgv <- df__ %>%
    filter(Surrogate %in% test_time_techs_on_lgv, Target_abbr == target_name) %>%
    mutate(
      mean_adv_success_rate_lgv = as.numeric(mean_adv_success_rate_lgv),
    ) %>%
    filter(mean_adv_success_rate < mean_adv_success_rate_lgv)
  worst_lgv_cells <- df___$Surrogate %in% df__worst_lgv$Surrogate
  worst_lgv_colors <- rep('black', nrow(df___))
  worst_lgv_colors[worst_lgv_cells] <- 'gray'
  table <- column_spec(table, 1+i, color = worst_lgv_colors)
  # table <- column_spec(table, 1+i, strikeout = worst_lgv_cells)
}
table

patch_table <- function(str) {
  str <- gsub('∞', "$\\infty$", str, fixed=T)
  # str <- sub('\\begin{table*}', '\\begin{table*} \n \\vskip 0.15in', str, fixed=T)
  # str <- sub('\\end{table*}', paste0('\\label{tab:main_results_L', norm_exported, '} \n \\vskip -0.1in \n \\end{table*}'), str, fixed=T)
  str <- sub('\\end{table*}', paste0('\\label{tab:main_results_L', norm_exported, '} \n \\end{table*}'), str, fixed=T)
  str <- gsub('100.0', '100', str, fixed=T)
  str <- gsub('SMALL', '\\tiny', str, fixed=T)
  str <- gsub('CLOSE', '}', str, fixed=T)
  str <- gsub('LGV + ', '', str, fixed=T)
  str <- gsub(' + ', '+', str, fixed=T)
  str <- gsub('LGV (ours)', '\\textbf{LGV (ours)}', str, fixed=T)
  return(str)
}

fileConn<-file(paste0("lgv/plots/table_main_results_L", norm_exported, ".tex"))
writeLines(patch_table(table), fileConn)
close(fileConn)


# compute the difference with several baselines (1 DNN, LGV, LGV-SWA)
df_ %>% 
  # filter(Surrogate %in% test_time_techs) %>%  # compare with test time techs
  # filter(Surrogate %in% test_time_techs_base) %>%  # compare with test time techs alone
  filter(Surrogate %in% test_time_techs_combinations) %>%  # compare with test time techs combinations
  #filter(Surrogate %in% test_time_techs_on_lgv) %>%  # LGV+techs
  #filter(Surrogate %in% ours) %>%  # compare with our surrogate
  ungroup() %>%
  left_join(
    df_ %>%
      filter(Surrogate == '1 DNN') %>%
      transmute(mean_adv_success_rate_1dnn = mean_adv_success_rate, Norm, `Target_abbr`)
  ) %>%
  left_join(
    df_ %>%
      filter(Surrogate == 'LGV (ours)') %>%
      transmute(mean_adv_success_rate_lgv = mean_adv_success_rate, Norm, `Target_abbr`)
  ) %>%
  left_join(
    df_ %>%
      filter(Surrogate == 'LGV-SWA') %>%
      transmute(mean_adv_success_rate_lgv_swa = mean_adv_success_rate, Norm, `Target_abbr`)
  ) %>%
  mutate(
    diff_success_rate_1dnn = mean_adv_success_rate_1dnn - mean_adv_success_rate,
    diff_success_rate_lgv = mean_adv_success_rate_lgv - mean_adv_success_rate,
    diff_success_rate_lgv_swa = mean_adv_success_rate_lgv_swa - mean_adv_success_rate,
  ) %>%
  # mean improvements
  #filter(Surrogate == 'LGV-SWA + RD') %>% summarise(mean(diff_success_rate_lgv), mean(diff_success_rate_lgv_swa))
  # summarise(mean(diff_success_rate_lgv), mean(diff_success_rate_lgv_swa))
  #group_by(Surrogate, Norm) %>% summarise(mean(diff_success_rate_lgv), pct_lgv_best=mean(diff_success_rate_lgv > 0)*100) %>% # analyse diff by technique
  # group_by(Surrogate, Target_abbr) %>% summarise(mean(diff_success_rate_lgv), pct_lgv_best=mean(diff_success_rate_lgv > 0)*100) %>% # analyse diff by technique
  View()


# baselines
df_tmp %>%
  filter(Surrogate_ %in% c('White-Box', '1 DNN', paste0(2:5, ' DNNs'), 'LGV (ours)'))



###################
#  OTHER ATTACKS  
###################

df <- read_csv('lgv/imagenet/attack_inter_arch_other_attacks.csv', show_col_types = FALSE) %>%
  full_join(read_csv('lgv/imagenet/attack_inter_arch_other_attacks_apgd.csv', show_col_types = FALSE)) %>%
  full_join(read_csv('lgv/imagenet/attack_inter_arch_other_attacks_square2.csv', show_col_types = FALSE)) %>%
  mutate(
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target),
    `Target_abbr` = recode_archs(arch_target, abbr=TRUE),
    TestTimeTechs_tmp = str_replace_all(str_replace_all(transferability_techniques, '_$', ''), '_', ' + '),
    TestTimeTechs = recode(TestTimeTechs_tmp,  # old=new
                           "ghost" = "GN",
                           "MI + ghost" = "GN + MI",
                           "MI + DI" = "DI + MI",
                           "MI + SGM" = "SGM + MI",
                           "DI + SGM" = "SGM + DI",
                           "MI + DI + SGM" = "SGM + DI + MI",
    ),
    Surrogate_ = case_when(
      surrogate_type=='ImageNet/pretrained' ~ 'White-Box',
      !is.na(TestTimeTechs) & grepl('/original$', model_surrogate) ~ TestTimeTechs, #paste0('1 DNN + ', TestTimeTechs),
      grepl('/original$', model_surrogate) ~ '1 DNN',
      grepl('/original/noisy/std_0.005_50models', model_surrogate) ~ 'RD',
      grepl('/PCA/dims_0', model_surrogate) ~ 'LGV-SWA',
      grepl('seed[0-9]+/noisy/random_ensemble_equivalent', model_surrogate) ~ 'LGV-SWA + RD',
      #surrogate_type=='dnn' & surrogate_size_ensembles==1 ~ '1 DNN',
      surrogate_type=='dnn' ~ paste0(surrogate_size_ensembles, ' DNNs'),
      !is.na(TestTimeTechs) & surrogate_type=='cSGD' ~ paste0('LGV + ', TestTimeTechs),
      surrogate_type=='cSGD' ~ 'LGV (ours)',
      TRUE ~ NA_character_
    )
  )
df_tmp <- df

df <- df %>%
  filter(seed == 0) %>%  #  TODO: tmp only 1 seed
  filter(!(surrogate_size_ensembles > 1 & grepl(' DNNs', Surrogate_))) %>%  # remove Deep Ensemble baseline 
  filter(Surrogate_ != 'White-Box')

# baselines = c('White-Box', '1 DNN', paste0(2:5, ' DNNs'))
baselines = c('1 DNN')
test_time_techs = c('MI',
                    'GN', 'GN + MI',
                    'DI', 'DI + MI',
                    'SGM', 'SGM + MI', 'SGM + DI', 'SGM + DI + MI')
test_time_techs_base = grep("\\+", test_time_techs, value = TRUE, invert = TRUE)
test_time_techs_combinations = grep("\\+", test_time_techs, value = TRUE)
test_time_techs_on_lgv <- paste0('LGV + ', test_time_techs)
ours <- c('RD', 'LGV-SWA', 'LGV-SWA + RD', 'LGV (ours)')
Surrogate_levels <- c(baselines, test_time_techs, ours, test_time_techs_on_lgv)
df <- df %>%
  mutate(
    Surrogate = factor(Surrogate_, levels = Surrogate_levels)
  )
if(df %>% filter(is.na(Surrogate), !is.na(Surrogate_)) %>% nrow()) warning('Lost levels in conversion!')

df_ <- df %>%
  group_by(attack_name, Norm, `Target_abbr`, Surrogate) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate * 100),
    sd_adv_success_rate = sd(adv_success_rate * 100),
    mean(transfer_rate* 100),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate),
  )
View(df_)
if(df_ %>% filter(n_seeds != 1) %>% nrow()) warning('Wrong number of seeds!') # TODO: 3 seeds for camera-ready


# compute the difference with several baselines (1 DNN, LGV, LGV-SWA)
df_ %>% 
  filter(Surrogate %in% c(test_time_techs, '1 DNN')) %>%  # compare with test time techs + 1 DNN
  #filter(Surrogate %in% test_time_techs) %>%  # compare with test time techs
  #filter(Surrogate %in% test_time_techs_base) %>%  # compare with test time techs alone
  #filter(Surrogate %in% test_time_techs_combinations) %>%  # compare with test time techs combinations
  #filter(Surrogate %in% test_time_techs_on_lgv) %>%  # LGV+techs
  #filter(Surrogate %in% ours) %>%  # compare with our surrogate
  ungroup() %>%
  left_join(
    df_ %>%
      filter(Surrogate == '1 DNN') %>%
      transmute(mean_adv_success_rate_1dnn = mean_adv_success_rate, Norm, `Target_abbr`)
  ) %>%
  left_join(
    df_ %>%
      filter(Surrogate == 'LGV (ours)') %>%
      transmute(mean_adv_success_rate_lgv = mean_adv_success_rate, Norm, `Target_abbr`)
  ) %>%
  left_join(
    df_ %>%
      filter(Surrogate == 'LGV-SWA') %>%
      transmute(mean_adv_success_rate_lgv_swa = mean_adv_success_rate, Norm, `Target_abbr`)
  ) %>%
  mutate(
    diff_success_rate_1dnn = mean_adv_success_rate_1dnn - mean_adv_success_rate,
    diff_success_rate_lgv = mean_adv_success_rate_lgv - mean_adv_success_rate,
    diff_success_rate_lgv_swa = mean_adv_success_rate_lgv_swa - mean_adv_success_rate,
  ) %>%
  # mean improvements
  #filter(Surrogate == 'LGV-SWA + RD') %>% summarise(mean(diff_success_rate_lgv), mean(diff_success_rate_lgv_swa))
  summarise(mean(diff_success_rate_lgv), mean(diff_success_rate_lgv_swa),  min(diff_success_rate_lgv_swa),  max(diff_success_rate_lgv_swa), pct_lgv_best=mean(diff_success_rate_lgv >= 0)*100, nb_cases_lgv_best=sum(diff_success_rate_lgv >= 0), nb_cases_total = n())
  #group_by(attack_name, Surrogate, Norm) %>% summarise(mean(diff_success_rate_lgv), pct_lgv_best=mean(diff_success_rate_lgv > 0)*100) %>% # analyse diff by technique
  # group_by(Surrogate, Target_abbr) %>% summarise(mean(diff_success_rate_lgv), pct_lgv_best=mean(diff_success_rate_lgv >= 0)*100) %>% # analyse diff by technique
  View()


# baselines
df_tmp %>%
  filter(Surrogate_ %in% c('White-Box', '1 DNN', paste0(2:5, ' DNNs'), 'LGV (ours)'))





#####################
#  NATURAL ACCURACY  
#####################

acc_target <- TRUE
filename <- if_else(acc_target, 'lgv/results/ImageNet/resnet50/cSGD/acc_test_targets.csv', 'lgv/results/ImageNet/resnet50/cSGD/acc_test.csv')
df <- read_csv(filename, show_col_types = FALSE) %>%
  mutate(
    seed = str_match(dir_models, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Architecture = recode_archs(str_remove(dir_models, 'ImageNet/pretrained/')),
    Name = recode_archs(str_remove(dir_models, 'ImageNet/pretrained/'), abbr = T),
    Method_ = case_when(
      model_type=='ImageNet/pretrained' ~ 'White-Box',
      grepl('/original$', dir_models) ~ '1 DNN',
      grepl('/original/noisy/std_0.005_50models', dir_models) ~ '1 DNN + RD',
      grepl('/PCA/dims_0', dir_models) ~ 'LGV-SWA',
      grepl('/cSGD/seed[0-9]+', dir_models) ~ 'LGV (ours)',
      TRUE ~ NA_character_
    ),
    Method = factor(Method_, levels = c('1 DNN', '1 DNN + RD', 'LGV-SWA', 'LGV (ours)'))
  )


if (acc_target) {
  df_ <- df %>%
    group_by(Architecture, Name) %>%
    summarise(
      mean_accuracy = mean(accuracy * 100),
      sd_accuracy = sd(accuracy * 100),
      mean_nll = mean(nll),
      sd_nll = sd(nll),
      `Number of models` = as.integer(mean(n_models)),
      n_seeds = n(),
      nb_ex = mean(nb_ex),
    )
  View(df_)
  df__ <- df_ %>%
    ungroup() %>%
    transmute(
      Name,
      Architecture,
      `Test Accuracy` = paste0(formatC(mean_accuracy, digits=2, format='f'), '%'),
      `Loss (NLL)` = formatC(mean_nll, digits=3, format='f')
      ) 
} else {
  df_ <- df %>%
    group_by(Method) %>%
    summarise(
      mean_accuracy = mean(accuracy * 100),
      sd_accuracy = sd(accuracy * 100),
      mean_nll = mean(nll),
      sd_nll = sd(nll),
      `Number of models` = as.integer(mean(n_models)),
      n_seeds = n(),
      nb_ex = mean(nb_ex),
    )
  df__ <- df_ %>%
    group_by(Method) %>%
    transmute(
      Method,
      `Test Accuracy` = paste0(formatC(mean_accuracy, digits=2, format='f'), '% SMALL ±', formatC(sd_accuracy, digits=2, format='f')),
      `Loss (NLL)` = paste0(formatC(mean_nll, digits=3, format='f'), ' SMALL ±', formatC(sd_nll, digits=3, format='f')),
      `Number of models`
    )
  View(df_)
}
View(df__)

debug_table <- F  # if T, print HTML, otherwise output LaTeX code
caption_ <- if_else(acc_target,
  "Natural accuracy and loss of target models computed on the test set.",
  "Natural accuracy and loss of surrogate models computed on the test set.")
table <- kbl(df__, booktabs = T, caption = caption_, format = ifelse(debug_table, 'html', 'latex'), align = c('ll', rep("r", 3-acc_target)), linesep = "") %>%
  kable_styling(latex_options = c("hold_position", "striped"), font_size=9) # "scale_down"
table

patch_table <- function(str) {
  if (acc_target) str <- sub('\\end{table}', '\n\\label{tab:target_natural_acc} \n \\end{table}', str, fixed=T)
  if (!acc_target) str <- sub('\\end{table}', '\n\\label{tab:surrogate_natural_acc} \n \\end{table}', str, fixed=T)
  str <- sub('\\begin{table}[!h]', '\\begin{table}[!ht] ', str, fixed=T)
  str <- gsub('SMALL', '\\tiny', str, fixed=T)
  return(str)
}

filename <- if_else(acc_target, 'lgv/plots/table_target_natural_acc.tex', 'lgv/plots/table_surrogate_natural_acc.tex')
fileConn<-file(filename)
writeLines(patch_table(table), fileConn)
close(fileConn)



#############
#  HP - LR  
#############


intraarch <- T # switch between intra and inter archs plots
# filename <- if_else(intraarch, 'lgv/results/ImageNet/resnet50/cSGD/HP/LR/attack.csv','lgv/results/ImageNet/resnet50/cSGD/HP/LR/attack_interarch.csv')
filename <- 'lgv/results/ImageNet/resnet50/cSGD/HP/LR/attack_interarch.csv'
df = read.csv(filename) %>%
  mutate(
    lr = if_else(grepl('original', model_surrogate), 0., as.numeric(str_match(model_surrogate, "HP/LR/(\\d+.?\\d*)/")[,2])), # 0 LR for original model
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target)
  ) %>%
  unite(id, c("Norm", "Dataset", "seed", "Target"), remove=FALSE) %>%
  mutate(Attack = Norm)
# check that we have 3 seeds per group
df %>%
  group_by(Norm, Dataset,`Target`, lr) %>%
  summarise(
    mean(adv_success_rate),
    mean(transfer_rate),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  ) %>% View()
# import natural accuracy on test/train set
df_acc <- read.csv('lgv/results/ImageNet/resnet50/cSGD/HP/LR/acc_test.csv') %>%
  mutate(Dataset = "Test") %>%
  union(
    read.csv('lgv/results/ImageNet/resnet50/cSGD/HP/LR/acc_train.csv') %>%
      mutate(Dataset = "Val.")
  ) %>%
  mutate(
    lr = if_else(grepl('original', dir_models), 0., as.numeric(str_match(dir_models, "HP/LR/(\\d+.?\\d*)/")[,2])),  # 0 LR for original model
    seed = str_match(dir_models, "seed(\\d+)")[,2],
  ) 

if(intraarch) {
  df_ <- df %>%
    filter(`Target` == 'ResNet-50')
} else {
  # original model already in the CSV
  df_ <- df
}

color_y_right <- 'orange'
breaks = c(1e+01, 1e+00, 1e-01, 1e-02, 1e-03, 0e+00)
p <- ggplot(df_, aes(x = lr, y = adv_success_rate, linetype=Dataset, colour=Attack, fill=Attack)) +
  stat_summary(aes(group = interaction(Attack, Dataset)), fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # mean +/- 1 std
  # geom='crossbar' / 'errorbar'
  geom_point(aes(shape = Attack), alpha=0.4, size=1.5) + 
  geom_point(aes(shape = Attack), size=0.1, colour='black') +
  xlab("Learning Rate (log)") +
  ylab("Success Rate") +
  # scale_x_continuous(trans=scales::pseudo_log_trans(base = 10, sigma = 0.0005), breaks = unique(df_$lr), labels = str_remove(unique(df_$lr), "\\..*(0+)$")) # labels = str_remove(unique(df_$lr), "\\..*(0+)$")
  scale_x_continuous(trans=scales::pseudo_log_trans(base = 10, sigma = 0.0003), breaks = breaks, labels = str_remove(breaks, "\\..*(0+)$")) # labels = str_remove(unique(df_$lr), "\\..*(0+)$")
  # scale_x_continuous(
  #   trans=scales::pseudo_log_trans(base = 10, sigma = 0.0005), 
  #   breaks = unique(df_$lr),
  #   labels = get_labelling_pseudo_log(label0 = 'Initial')
  # ) +
  # scale_x_continuous(trans='log10', breaks = unique(df$lr)) + # limits = c(0, 0.1)
  # tune sigma to looks good with plot(pseudo_log_trans(base = 10, sigma = 0.001), xlim = c(0, 0.1))
  #theme_light()

# add secondary axis with standard accuracy
if(intraarch) {
  # p <- p + stat_summary(data = df_acc, aes(x = lr, y = accuracy), fun.data = "mean_se", geom = "smooth", se = TRUE, col = color_y_right, fill= color_y_right, linetype='dotted', alpha=0.4, inherit.aes = FALSE) +  # mean +/- 1 std
  p <- p + stat_summary(data = df_acc, aes(x = lr, y = accuracy, linetype=Dataset), fun.data = "mean_se", geom = "smooth", se = TRUE, colour = color_y_right, fill = color_y_right, alpha=0.4, show.legend=FALSE) + # linetype='dotted' # mean +/- 1 std
  #geom_line(data = df_acc, aes(x = lr, y = accuracy, linetype=''), col = color_y_right, inherit.aes = FALSE) + # add test accuracy
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                                     sec.axis = dup_axis(
                                       name = "Natural Surrogate Accuracy",
                                     ) ) +
  theme(axis.line.y.right = element_line(color = color_y_right), # color secondary axis
          axis.ticks.y.right = element_line(color = color_y_right),
          axis.text.y.right = element_text(color = color_y_right),
          axis.title.y.right = element_text(color = color_y_right),
          axis.text.x = element_text(size=13))
} else {
  p <- p + scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  facet_wrap( ~ `Target`, nrow=2) +
  theme(
    axis.text.x = element_text(size=10)
  )
}
show(p)
filename_export <- if_else(intraarch, 'lgv/plots/hp_lr.pdf','lgv/plots/hp_lr_interarch.pdf')
ggsave(filename=filename_export, width=17.43*(2-intraarch/1.5), height=13.79, units="cm", device=cairo_pdf)  # size of template example figure


# best LR
df %>%
  group_by(Norm, Dataset, Target, lr) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate),
  ) %>%
  filter(mean_adv_success_rate == max(mean_adv_success_rate)) %>%
  View()


# plot losses
p <- ggplot(df_, aes(x = lr, y = loss_adv, linetype=Dataset, colour=Norm, fill=Norm)) + 
  stat_summary(aes(group = interaction(Norm, Dataset)), fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # mean +/- 1 std
  # geom='crossbar' / 'errorbar'
  geom_point(aes(shape = Norm), alpha=0.4, size=1.5) + 
  geom_point(aes(shape = Norm), size=0.1, colour='black') +
  xlab("Learning Rate (log)") +
  ylab("Adversarial Target Loss") +
  scale_x_continuous(trans=scales::pseudo_log_trans(base = 10, sigma = 0.0003), breaks = breaks, labels = str_remove(breaks, "\\..*(0+)$")) + # labels = str_remove(unique(df_$lr), "\\..*(0+)$")
  stat_summary(data = df_acc, aes(x = lr, y = nll, linetype=Dataset), fun.data = "mean_se", geom = "smooth", se = TRUE, col = color_y_right, fill = color_y_right, alpha=0.4) + # linetype='dotted' # mean +/- 1 std
    scale_y_continuous(
                       sec.axis = dup_axis(
                         name = "Natural Loss",
                       ) ) +
    theme(axis.line.y.right = element_line(color = color_y_right), # color secondary axis
          axis.ticks.y.right = element_line(color = color_y_right),
          axis.text.y.right = element_text(color = color_y_right),
          axis.title.y.right = element_text(color = color_y_right),
          axis.text.x = element_text(size=13))
show(p)


df_val <- df %>% filter(Dataset == 'Val.')
df_test <- df %>% filter(Dataset == 'Test')
ggplot(df_val, aes(x = lr, y = adv_success_rate, shape = Dataset, colour=Norm, group = interaction(seed, Norm, Dataset))) + 
  geom_point(alpha=0.8) + 
  geom_line(aes(color=Norm, group=id)) +
  xlab("Learning Rate") +
  ylab("Success Rate")




ggplot(df, aes(x = lr, y = transfer_rate, shape = Dataset, colour=Norm, group = interaction(Norm, Dataset))) + 
  stat_summary(fun = mean, geom = "smooth", se = TRUE) +
  geom_point() +
  xlab("Learning Rate") +
  ylab("Transfer Rate")


df_grouped <- df %>%
  group_by(Norm, Dataset, lr) %>%
  summarise(
    mean_sr = mean(adv_success_rate),
    sd_sr = sd(adv_success_rate),
  ) %>%
  mutate(
    low = mean_sr - sd_sr,
    high = mean_sr + sd_sr
  )
ggplot(df_grouped, aes(x = lr, y = mean_sr, group = interaction(Norm, Dataset))) + 
  geom_line(col='red') + 
  geom_point() + 
  geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.1) +
  xlab("Learning Rate")


#######################
#  RQ0 - NOISY MODELS
######################

main_plot <- F

filename <- 'lgv/results/ImageNet/resnet50/RQ0/attack_noisy_original_interarch.csv'
label0 <- "Original"

filename <- 'lgv/results/ImageNet/resnet50/RQ2/attack_noisy_swa_interarch.csv'
label0 <- "LGV-SWA"

df = read.csv(filename) %>%
  { if(label0=="LGV-SWA") filter(., row_number() > 3*2*8*2) else . } %>% # remove duplicated run on the original model
  mutate(
    std =  case_when(
      grepl('/original$', model_surrogate) ~ 0., # original model
      grepl('/PCA/dims_0$', model_surrogate) ~ 0., # LGV-SWA model
      TRUE ~ as.numeric(str_match(model_surrogate, "/noisy/std[/_](\\d+.?\\d*)")[,2])
    ),
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target)
  )

# check that we have 3 seeds
df %>%
  # filter(surrogate_size_ensembles %in% c(1, 50)) %>%  #  uncomment if only shows original+50 noisy models (and no 10 models ensemble used for HP tuning)
  filter(surrogate_size_ensembles <= 10) %>%
  group_by(Norm, Dataset, `Target`, surrogate_size_ensembles, std) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate) * 100,
    mean(transfer_rate) * 100,
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  ) %>% View()

# filter rows for HP search: 1 (original) or 10 models (noisy ensemble)
df_hp <- df %>% 
  filter(surrogate_size_ensembles <= 10)



x_breaks = c(0L, 1e-02, 1e-03, 1e-04)
ggplot(df_hp, aes(x = std, y = adv_success_rate, shape = Dataset, linetype=Dataset, colour=Norm, fill=Norm, group = interaction(Norm, Dataset))) + 
  stat_summary(fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # mean +/- 1 std
  # geom='crossbar' / 'errorbar'
  geom_point(alpha=0.6, size=1.5, show.legend=FALSE) +
  geom_point(size=0.1, colour='black', show.legend=FALSE) +
  xlab("Standard Deviation of Weights Gaussian Noise (log)") +
  ylab("Success Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  # scale_x_continuous(trans='log10', breaks = unique(df_hp$std)) +
  scale_x_continuous(
    trans=scales::pseudo_log_trans(base = 10, sigma = 1e-05),
    breaks = x_breaks,
    # labels = str_replace(x_breaks, '0e+00', '0')
    labels = get_labelling_pseudo_log(label0 = label0)
  ) +
  # theme_light() +
  facet_wrap( ~ `Target`, nrow=2, scales = 'free_y')
ggsave(filename=paste0('lgv/plots/rq0_white_noise_weights_', tolower(label0), '.pdf'), width=17.43*2, height=13.79, units="cm", device=cairo_pdf)  # size of template example figure


#########################
#  RQ0 - NOISY GRADIENTS
#########################

filename <- 'lgv/results/ImageNet/resnet50/RQ0/attack_grad_noise_original_interarch.csv'
df = read.csv(filename) %>%
  mutate(
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target)
  )

# check that we have 3 seeds
df_mean <- df %>%
  group_by(Norm, Dataset, `Target`, grad_noise_std) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate) * 100,
    mean(transfer_rate) * 100,
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  )
View(df_mean)

x_breaks = c(0L, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02)
ggplot(df, aes(x = grad_noise_std, y = adv_success_rate, shape = Dataset, linetype=Dataset, colour=Norm, fill=Norm, group = interaction(Norm, Dataset))) + 
  stat_summary(fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # mean +/- 1 std
  # geom='crossbar' / 'errorbar'
  geom_point(alpha=0.6, size=1.5, show.legend=FALSE) +
  xlab("Standard Deviation of Attack Gradients Gaussian Noise (log)") +
  ylab("Success Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_x_continuous(
    trans=scales::pseudo_log_trans(base = 10, sigma = 1e-08), 
    breaks = x_breaks,
    labels = get_labelling_pseudo_log(label0 = 'Vanilla\nI-FGSM')
    ) +
  # theme_light() +
  theme(
    # legend.position = "bottom",
    axis.text.x = element_text(vjust = 0, size = 10),
    #legend.text = element_text(vjust = 0)
  ) +
  facet_wrap( ~ `Target`, nrow=2, scales = 'free_y')
ggsave(filename='lgv/plots/rq0_white_noise_gradients.pdf', width=17.43*2, height=13.79, units="cm", device=cairo_pdf)  # size of template example figure

# best grad_noise_std *for each target arch* on the val set (1 std for 1 target and norm)
df_mean %>%
  filter(Dataset == "Val.") %>%
  group_by(`Target`, Norm) %>%
  filter(mean_adv_success_rate == max(mean_adv_success_rate)) %>%
  transmute(Dataset, `Target`, Norm, grad_noise_std, mean_adv_success_rate, n_seeds) %>%
  arrange(Dataset, `Target`, Norm) %>%
  left_join( # add corresponding test success 
    df_mean %>% 
      ungroup() %>%
      filter(Dataset == "Test") %>%
      transmute(mean_adv_success_rate_TEST=mean_adv_success_rate, `Target`, Norm, grad_noise_std)
  ) %>%
  left_join( # add corresponding original model 
    df_mean %>% 
      ungroup() %>%
      filter(Dataset == "Test", grad_noise_std==0) %>%
      transmute(mean_adv_success_rate_TEST_original=mean_adv_success_rate, `Target`, Norm)
  ) %>%
  mutate(increase_success_rate_test = mean_adv_success_rate_TEST - mean_adv_success_rate_TEST_original) %>%
  #ungroup() %>% summarise(mean(increase_success_rate_test))
  View()

# best grad_noise_std *for all target arch* on the val set (1 std for all targets and 1 norm)
df_hp <- df %>% 
  filter(Dataset == "Val.")

df_hp %>% 
  left_join( # add original
    df_hp %>% 
      filter(grad_noise_std==0) %>%
      transmute(adv_success_rate_original = adv_success_rate, Dataset, Norm, `Target`, seed)
  ) %>%
  ungroup() %>%
  mutate(
    adv_success_rate_diff = adv_success_rate - adv_success_rate_original
  ) %>%
  group_by(Norm, grad_noise_std) %>%
  summarise(
    mean_adv_success_rate_diff = mean(adv_success_rate_diff),
    n_ = n(),
  ) %>% View()



####################################
#  RQ1 - INTERPOLATION SWA-ORIGINAL
####################################

filename <- 'lgv/results/ImageNet/resnet50/RQ1/attack_interpolation_SWA_original_interarch.csv'
df = read.csv(filename) %>%
  mutate(
    alpha = as.numeric(str_match(model_surrogate, "/alpha_(\\-?\\d+.?\\d*)")[,2]),
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target)
  )

df_acc = read.csv("lgv/results/ImageNet/resnet50/RQ1/accuracy_interpolation_SWA_original.csv") %>%
  mutate(
    alpha = as.numeric(str_match(dir_models, "/alpha_(\\-?\\d+.?\\d*)")[,2]),
    seed = str_match(dir_models, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test")
  )

# check that we have 1 seed
df_ <- df %>%
  group_by(Norm, Dataset, `Target`, alpha) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate) * 100,
    mean(transfer_rate) * 100,
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  )
View(df_)

# best alphas
df_ %>%
  group_by(Norm, Dataset, `Target`) %>%
  # filter(Target %in% ARCHS_RESNET_FAMILY) %>%
  # group_by(Norm, Dataset) %>%
  filter(mean_adv_success_rate == max(mean_adv_success_rate)) %>%
  View()

df__ <- df %>%
  mutate(
    Loss = paste0('Adv. Target ', Target)
  ) %>%
  add_row(
    df %>% 
      distinct(alpha, surrogate_loss_original_ex) %>%
      mutate(
        Loss = 'Natural Surrogate'
      )
  )


plot_loss <- TRUE  # produce either the plot of losses or accuracy/success rate
main_plot <- T

color_y_right='orange'
if(plot_loss & main_plot) p <- ggplot(df %>% filter(`Target` %in% ARCHS_MAIN_PLOT), aes(x = alpha, y = loss_adv, colour=Norm, group = interaction(`Target`, Norm), linetype=`Target`))
if(plot_loss & !main_plot) p <- ggplot(df, aes(x = alpha, y = loss_adv, colour=Norm, group = Norm))
if(!plot_loss) p <- ggplot(df, aes(x = alpha, y = adv_success_rate, colour=Norm, group = Norm))
p <- p +
  geom_line(alpha=0.7, size=1.2) +
  xlab("Interpolation Coefficient") +
  ylab(ifelse(plot_loss, "Loss", "Success Rate")) +
  # geom_line(data = df_acc, aes(x = alpha, y = nll, colour=dataset, group=dataset), alpha=0.7, linetype='dotted', inherit.aes=FALSE) + # loss computed on all test
  geom_line(data=df %>% distinct(alpha, surrogate_loss_original_ex), aes(x = alpha, y = surrogate_loss_original_ex), color=color_y_right, size=1.2, alpha=1, linetype='dashed', inherit.aes = FALSE) +  # %>% mutate(surrogate_loss_original_ex=replace(surrogate_loss_original_ex, is.infinite(surrogate_loss_original_ex), 99)
  # scale_y_continuous(sec.axis = sec_axis(
  #                      ~ .,
  #                      name = "Natural Test Loss",
  #                    ) ) +
  #   theme(axis.line.y.right = element_line(color = color_y_right), # color secondary axis
  #       axis.ticks.y.right = element_line(color = color_y_right),
  #       axis.text.y.right = element_text(color = color_y_right),
  #       axis.title.y.right = element_text(color = color_y_right)) +
  scale_x_continuous(breaks = c(-1, -.5, 0, 0.5, 1, 1.5, 2), labels = c(-1, '', '0\nLGV-SWA', '', '1\nInitial DNN', '', 2)) + # limits = c(0, 0.1)
  # scale_y_continuous(limits = c(0, 8.2)) +
  { if(!main_plot) facet_wrap( ~ `Target`, nrow=2) }
  # theme_light()
show(p)
filename_export <- case_when(
  plot_loss & main_plot ~ 'lgv/plots/rq1_interpol_swa_original_loss_main.pdf',
  plot_loss & !main_plot ~ 'lgv/plots/rq1_interpol_swa_original_loss_all.pdf',
  !plot_loss ~ 'lgv/plots/rq1_interpol_swa_original_successrate.pdf',
  TRUE ~ NA_character_ 
)
ggsave(filename=filename_export, width=17.43*(2-main_plot), height=13.79, units="cm", device=cairo_pdf)  # size of template example figure
ggsave(filename=str_replace(filename_export, '.pdf','.png'), width=17.43*(2-main_plot), height=13.79, units="cm")


#######################
#  RQ0 - NOISY MODELS
######################

main_plot <- F

filename <- 'lgv/results/ImageNet/resnet50/RQ0/attack_noisy_original_interarch.csv'
label0 <- "Original"

filename <- 'lgv/results/ImageNet/resnet50/RQ2/attack_noisy_swa_interarch.csv'
label0 <- "LGV-SWA"

df = read.csv(filename) %>%
  { if(label0=="LGV-SWA") filter(., row_number() > 3*2*8*2) else . } %>% # remove duplicated run on the original model
  mutate(
    std =  case_when(
      grepl('/original$', model_surrogate) ~ 0., # original model
      grepl('/PCA/dims_0$', model_surrogate) ~ 0., # LGV-SWA model
      TRUE ~ as.numeric(str_match(model_surrogate, "/noisy/std[/_](\\d+.?\\d*)")[,2])
    ),
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target)
  )

# check that we have 3 seeds
df %>%
  # filter(surrogate_size_ensembles %in% c(1, 50)) %>%  # TODO: uncomment if only shows original+50 noisy models (and no 10 models ensemble used for HP tuning)
  group_by(Norm, Dataset, `Target`, surrogate_size_ensembles, std) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate) * 100,
    mean(transfer_rate) * 100,
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  ) %>% View()

# filter rows for HP search: 1 (original) or 10 models (noisy ensemble)
df_hp <- df %>% 
  filter(surrogate_size_ensembles <= 10)



x_breaks = c(0L, 1e-02, 1e-03, 1e-04)
ggplot(df_hp, aes(x = std, y = adv_success_rate, shape = Dataset, linetype=Dataset, colour=Norm, fill=Norm, group = interaction(Norm, Dataset))) + 
  stat_summary(fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # mean +/- 1 std
  # geom='crossbar' / 'errorbar'
  geom_point(alpha=0.6, size=1.5, show.legend=FALSE) +
  geom_point(size=0.1, colour='black', show.legend=FALSE) +
  xlab("Standard Deviation of Weights Gaussian Noise (log)") +
  ylab("Success Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  # scale_x_continuous(trans='log10', breaks = unique(df_hp$std)) +
  scale_x_continuous(
    trans=scales::pseudo_log_trans(base = 10, sigma = 1e-05),
    breaks = x_breaks,
    # labels = str_replace(x_breaks, '0e+00', '0')
    labels = get_labelling_pseudo_log(label0 = label0)
  ) +
  # theme_light() +
  facet_wrap( ~ `Target`, nrow=2, scales = 'free_y')
ggsave(filename=paste0('lgv/plots/rq0_white_noise_weights_', tolower(label0), '.pdf'), width=17.43*2, height=13.79, units="cm", device=cairo_pdf)  # size of template example figure


#############################################################
#  HP - TRANSLATION ALPHA HP for "1 DNNN + gamma*(LGV - LGV-SWA)"
#############################################################

filename <- 'lgv/results/ImageNet/resnet50/RQ1/attack_HP_alpha_translated_lgv_to_dnn_interarch.csv'
df = read.csv(filename) %>%
  mutate(
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    alpha = as.numeric(str_match(model_surrogate, "HP/alpha/(\\d+.?\\d*)/")[,2]),
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target)
  )

# check that we have 3 seeds
df_mean <- df %>%
  group_by(Norm, Dataset, `Target`, alpha) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate) * 100,
    mean(transfer_rate) * 100,
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  )
View(df_mean)

x_breaks = c(1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0)
ggplot(df, aes(x = alpha, y = adv_success_rate, shape = Dataset, linetype=Dataset, colour=Norm, fill=Norm, group = interaction(Norm, Dataset))) + 
  stat_summary(fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # mean +/- 1 std
  # geom='crossbar' / 'errorbar'
  geom_point(alpha=0.6, size=1.5, show.legend=FALSE) +
  xlab("Scale of LGV Deviations applied to 1 DNN") +
  ylab("Success Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_x_continuous(
  #   trans=scales::pseudo_log_trans(base = 10, sigma = 1e-08), 
    breaks = x_breaks,
  #   labels = get_labelling_pseudo_log(label0 = 'Vanilla\nI-FGSM')
  ) +
  theme(
    # legend.position = "bottom",
    axis.text.x = element_text(vjust = 0, size = 12),
    #legend.text = element_text(vjust = 0)
  ) +
  facet_wrap( ~ `Target`, nrow=2, scales = 'free_y')
ggsave(filename='lgv/plots/rq1_hp_alpha_translate_1dnn.pdf', width=17.43*2, height=13.79, units="cm", device=cairo_pdf)  # size of template example figure


# analyse
df_mean %>% 
  group_by(Norm, Dataset, `Target`) %>%
  mutate(diff_optimal=mean_adv_success_rate-max(mean_adv_success_rate)) %>%
  filter(alpha==1, Dataset=="Test") %>%
  # ungroup() %>% summarize(mean(diff_optimal))
  View()



####################################
#  RQ1 - DISK IN FEATURE SPACE
####################################

main_plot <- T  # true: LGV, Initial DNN

for (i in 1:(7-main_plot*6)) {
  if (i==1){
    filename <- 'lgv/results/ImageNet/resnet50/RQ1/feature_space/disk_LGV_original.csv'
    label0 <- 'LGV'
    label1 <- 'Initial DNN'
    
  } else if (i==2) {
    filename <- 'lgv/results/ImageNet/resnet50/RQ1/feature_space/disk_SWA_original.csv'
    label0 <- 'LGV-SWA'
    label1 <- 'Initial DNN'
    
  } else if (i==3) {
    filename <- 'lgv/results/ImageNet/resnet50/RQ1/feature_space/disk_LGV_SWA.csv'
    label0 <- 'LGV'
    label1 <- 'LGV-SWA'
    
  } else if (i==4) {
    filename <- 'lgv/results/ImageNet/resnet50/RQ1/feature_space/disk_original_noisy.csv'
    label0 <- 'Initial DNN'
    label1 <- 'Initial noisy DNN'
    
  } else if (i==5) {
    filename <- 'lgv/results/ImageNet/resnet50/RQ1/feature_space/disk_LGVindiv0_original.csv'
    filename <- 'lgv/results/ImageNet/resnet50/RQ1/feature_space/disk_LGVindiv1_original.csv'
    filename <- 'lgv/results/ImageNet/resnet50/RQ1/feature_space/disk_LGVindiv2_original.csv'
    label1 <- 'LGV individual'
    label0 <- 'Initial DNN'
    
  } else if (i==6) {
    filename <- 'lgv/results/ImageNet/resnet50/RQ1/feature_space/disk_LGVindiv0_SWA.csv'
    label1 <- 'LGV individual'
    label0 <- 'LGV-SWA'
    
  } else if (i==7) {
    filename <- 'lgv/results/ImageNet/resnet50/RQ1/feature_space/disk_LGV_LGVindiv0.csv'
    label1 <- 'LGV'
    label0 <- 'LGV individual'
  }
  
  df = read.csv(filename) %>%
    mutate(
      Model_ = case_when(
        grepl('/original/noisy/', model) ~ 'Surrogate Initial DNN + RD',
        grepl('/original/', model) ~ 'Surrogate Initial DNN',
        grepl('/PCA/dims_0/', model) ~ 'Surrogate LGV-SWA',
        grepl('cSGD/seed[0-9]$', model) ~ 'Surrogate LGV',
        grepl('cSGD/seed0/iter-[0-9]+.pt$', model) ~ 'Surrogate LGV indiv.',
        grepl('ImageNet/pretrained', model) ~ paste0('Target ', recode_archs(str_replace_all(model, 'ImageNet/pretrained/', ''))), # as.character
        TRUE ~ NA_character_
      ),
      Model = factor(Model_, levels = c('Surrogate LGV', 'Surrogate LGV indiv.', 'Surrogate LGV-SWA', 'Surrogate Initial DNN', 'Surrogate Initial DNN + RD', paste0('Target ', ALL_ARCHS))),
      Model_abbr_ = abbreviate_archs(Model),
      Model_abbr = factor(Model_abbr_, levels = c('Surrogate LGV', 'Surrogate LGV indiv.', 'Surrogate LGV-SWA', 'Surrogate Initial DNN', 'Surrogate Initial DNN + RD', paste0('Target ', abbreviate_archs(ALL_ARCHS)))),
      Architecture_ = case_when(
        grepl('/ImageNet/resnet50/', model) ~ 'resnet50',
        grepl('ImageNet/pretrained', model) ~ str_replace_all(model, 'ImageNet/pretrained/', ''),
        TRUE ~ NA_character_
      ),
      Architecture = recode_archs(Architecture_), 
      type_model_ = case_when(
        # Model == 'Surrogate LGV' ~ "LGV Surrogate",
        # Model == 'Surrogate LGV-SWA' ~ "LGV-SWA Surrogate",
        # Model == 'Surrogate Initial DNN' ~ "Initial DNN Surrogate",
        type_model == 'surrogate' ~ as.character(Model),
        type_model == 'target' ~ 'Target',
        TRUE ~ NA_character_
      ),
      `Model Type` = factor(type_model_, levels = c('Surrogate LGV', 'Surrogate Initial DNN', 'Surrogate LGV-SWA', 'Surrogate Initial DNN + RD', 'Target')),
      Norm = recode(norm, "2" = "L2", "Inf" = "L∞"),
      success_rate = 1 - adv_accuracy,
      r <- sqrt(x1^2 + x2^2),
      theta <- atan(x2/x1)
    )

  df_points <- read_csv(sub('.csv', '__ref_xadv.csv', filename), show_col_types = FALSE) %>%
    group_by(model, type_model) %>%
    summarise(
      x1 = mean(x1),
      x2 = mean(x2),
    ) %>%
    ungroup() %>%
    mutate(
      Point_ = case_when(
        grepl('/original/noisy/', model) ~ 'Surrogate Initial DNN + Noise',
        grepl('/original/', model) ~ 'Surrogate Initial DNN',
        grepl('/PCA/dims_0/', model) ~ 'Surrogate LGV-SWA',
        grepl('cSGD/seed[0-9]$', model) ~ 'Surrogate LGV',
        grepl('cSGD/seed0/iter-[0-9]+.pt$', model) ~ 'LGV indiv.',
        TRUE ~ NA_character_
      ),
      Point = paste0('Adv. ', Point_)
    ) %>%
    add_row(x1=0, x2=0, Point='Natural', type_model='x')
  
  
  if(main_plot) df_ <- df %>% filter(Architecture %in% ARCHS_MAIN_PLOT)
  if(!main_plot) df_ <- df
  
  ggplot(df_) + 
    geom_raster(aes(x1, x2, fill=adv_loss), interpolate = TRUE) + # success_rate adv_loss
    geom_contour(aes(x=x1, y=x2, z=adv_loss), colour="black") + 
    geom_circle(data=data.frame(x0 = 0, y0 = 0, r = 3.), aes(x0 = x0, y0 = y0, r = r), inherit.aes = FALSE, color='white', size=0.5) +
    coord_fixed() + # circularity
    scale_fill_viridis_c(name = "Loss (log)", trans='log', option = "inferno", breaks=c(10, 1, 0.1), labels=c('10', '1.0', '0.1')) +
    geom_point(data = df_points, aes(x1, x2, shape=type_model), color='white', size=2.5) +
    scale_shape_manual(name = "Point", values = c("x" = 1, "surrogate_1" = 0, "surrogate_2" = 2), guide = 'none') +
    facet_wrap( ~ Model, nrow=2-main_plot) +
    theme_no_axes(base.theme = theme_bw(base_size = 16) + theme(strip.text = element_text(size = 15)))

  filename_export <- if_else(main_plot, 
                             paste0('lgv/plots/feature_space/disk_', gsub(' ', '_', label0), '_',  gsub(' ', '_', label1), '_main.pdf'),
                             paste0('lgv/plots/feature_space/disk_', gsub(' ', '_', label0), '_',  gsub(' ', '_', label1), '_all.pdf'))
  ggsave(filename=filename_export, width=17.43*2, height=13.79/(0.9+main_plot*.8), units="cm", device=cairo_pdf)
  if(main_plot) ggsave(filename=sub('.pdf', '.png', filename_export), width=17.43*2, height=13.79/(0.9+main_plot*.8), units="cm")
}


############################
#  RQ1 - INDIVIDUAL MODELS
############################

main_plot <- T

filename <- 'lgv/imagenet/attack_individual_model.csv'
df = read.csv(filename) %>%
  mutate(
    model_index = as.integer(str_match(model_surrogate, "/iter-(\\d+)\\.pt")[,2]),
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target),
    Surrogate_ = case_when(
      grepl('/original$', model_surrogate) ~ 'Initial\nDNN',
      grepl('/seed[0-9]+$', model_surrogate) ~ 'LGV',
      grepl('/iter-(\\d+)\\.pt$', model_surrogate) ~ 'Indiv.\nLGV',
      TRUE ~ NA_character_
    ),
    Surrogate = factor(Surrogate_, levels = c('Initial\nDNN', 'Indiv.\nLGV', 'LGV'))
  )

# check that we have 3 seeds
df_ <- df %>%
  group_by(Norm, Dataset, `Target`, Surrogate, model_index) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate) * 100,
    mean(transfer_rate) * 100,
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  )
View(df_)
if(df_ %>% filter(n_seeds != 3) %>% nrow()) warning('Wrong number of seeds!')

df__ <- df %>% 
  filter(Surrogate %in% c('Indiv.\nLGV', 'Initial\nDNN')) %>%
  mutate(
    model_index = if_else(Surrogate == 'Initial\nDNN', 0L, model_index)
  ) %>%
  add_row( # duplicate Original to have a horizontal line
    df %>% filter(Surrogate == 'Initial\nDNN') %>% mutate(model_index=40L)
  )

if(main_plot) df__ <- df__ %>% filter(Target=='ResNet-50')

ggplot(df__, aes(x = model_index, y = adv_success_rate, colour=Norm, linetype=Surrogate, group = interaction(Norm, Surrogate), fill=Norm)) +
  geom_point(data=df__ %>% filter(Surrogate=='Indiv.\nLGV'), alpha=0.3, size=0.8) + 
  stat_summary(fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # aes=aes(linetype='solid')
  xlab("LGV Model Index") +
  ylab("Success Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  {if(!main_plot) facet_wrap( ~ `Target`, nrow=2, scales = 'free_y')}
if(!main_plot) ggsave(filename='lgv/plots/rq1_lgv_individual_models.pdf', width=17.43*2, height=13.79, units="cm", device=cairo_pdf)  # size of template example figure
if(main_plot) ggsave(filename='lgv/plots/rq1_lgv_individual_models_main.png', width=17.43, height=13.79, units="cm")



####################################
#  RQ1 - RANDOM DIRECTIONS FROM SWA
####################################

main_plot <- T
norm_type_main_plot <- Inf # 2  # attack norm of the main plot

filename <- 'lgv/results/ImageNet/resnet50/RQ1/attack_random_1D_interarch.csv'
df = read.csv(filename) %>%
  mutate(
    Surrogate = case_when(
      grepl('/original/', model_surrogate) ~ '1 DNN',
      grepl('/PCA/dims_0/', model_surrogate) ~ 'LGV-SWA',
      grepl('/cSGD/seed[0-9]+/random_1D/', model_surrogate) ~ 'LGV indiv.',
      TRUE ~ NA_character_
    ),
    distance_1d = as.numeric(str_match(model_surrogate, "/norm_(\\d+.?\\d*)$")[,2]),
    dim_id = as.numeric(str_match(model_surrogate, "/dim_(\\d+)/")[,2]),
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target),
    `Loss Type` = paste0("Adversarial Loss on ", Target, " Target"),
    Loss = loss_adv,
  )

df %>%
  # group_by(Norm, Dataset, `Target`, dim_id, distance_1d, Surrogate) %>%
  group_by(Norm, Dataset, `Target`, distance_1d, Surrogate) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate) * 100,
    mean(transfer_rate) * 100,
    mean(surrogate_loss_original_ex),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  ) %>% View()

df_ <- df %>% add_row(
  df %>%
    filter(Target=='ResNet-50', norm_type == norm_type_main_plot) %>% # filter to avoid duplicates of natural loss
    mutate(
    `Loss Type` = "Natural Loss on ResNet-50 Surrogate",
    Loss = surrogate_loss_original_ex,
  )
) %>%
  mutate(
    `Loss Type` = factor(`Loss Type`, levels=c('Natural Loss on ResNet-50 Surrogate',
                                               "Adversarial Loss on ResNet-50 Target",
                                               "Adversarial Loss on ResNet-152 Target",
                                               "Adversarial Loss on ResNext-50 Target",
                                               "Adversarial Loss on WideResNet-50 Target",
                                               "Adversarial Loss on DenseNet-201 Target",
                                               "Adversarial Loss on VGG19 Target",
                                               "Adversarial Loss on Inception v1 Target",
                                               "Adversarial Loss on Inception v3 Target"
    ))
  )


# main plot: 1 norm, 1 target 
# all plots: 2 norms, 1 target per subplot
df__ <- df_ #%>% filter(norm_type == norm_type_main_plot)  # filter to avoid duplicates of natural loss
if (main_plot) df__ <- df_ %>% filter(Target %in% c('ResNet-50'), norm_type == norm_type_main_plot)

# main plot: only Linf
if(main_plot) p <- ggplot(df__, aes(x = distance_1d, y = Loss, colour=Surrogate, group = interaction(`Loss Type`, `Target`, Surrogate, dim_id), shape = Surrogate)) # linetype=Surrogate
if(!main_plot) p <- ggplot(df__, aes(x = distance_1d, y = Loss, colour=Surrogate, group = interaction(`Loss Type`, Norm, `Target`, Surrogate, dim_id), shape = Norm, linetype=Norm))

p <- p +
  geom_line(size=0.4, alpha=0.5) +
  xlab("Distance along Random Directions") +
  ylab("Loss") +
  scale_color_brewer(palette = 'Accent') +  # color palette for surrogate Accent
  guides(color = guide_legend(override.aes = list(size = 1.5))) +
  facet_wrap( ~ `Loss Type`, nrow=3, scales="free_y") 

if(!main_plot) p <- p + theme(strip.text.x = element_text(size=12))
show(p)
filename_export <- if_else(main_plot, 
                           paste0('lgv/plots/rq1_random_directions_from_swa_main_L', norm_type_main_plot, '.pdf'),
                           'lgv/plots/rq1_random_directions_from_swa_all.pdf')
ggsave(filename=filename_export, width=17.43*(2-main_plot), height=(1.5-main_plot*0.5)*13.79, units="cm", device=cairo_pdf)  # size of template example figure
ggsave(filename=str_replace(filename_export, '.pdf', '.png'), width=17.43*(2-main_plot), height=1.5*13.79, units="cm")  # size of template example figure


#############################
#  RQ1 - PCA PROJECTION DIMS
#############################

main_plot <- T # switch between intra and inter archs plots
main_plot_arch <- 'ResNet-50'
main_plot_arch <- 'Inception v3'
filename <- 'lgv/results/ImageNet/resnet50/RQ1/attack_proj_pca_dims_interarch.csv'
df = read.csv(filename) %>%
  mutate(
    n_dims = case_when(
      grepl('seed(\\d+)$', model_surrogate) ~ 40, # original model
      TRUE ~ as.numeric(str_match(model_surrogate, "/dims_(\\d+)")[,2])
    ),
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target)
  ) %>%
  filter(!grepl('original', model_surrogate)) %>%  # we remove original model: maybe used later
  unite(id, c("Norm", "seed"), remove=FALSE)

if(main_plot) df <- df %>% filter(Target == !!main_plot_arch) %>% droplevels()
if(!main_plot) df <- df %>% filter(Target %in% ALL_ARCHS) %>% droplevels()

df_pca <- read.csv('lgv/results/ImageNet/resnet50/cSGD/seed0/PCA/metrics_pca.csv') %>%
  mutate(seed = '0') %>%
  union(
    read.csv('lgv/results/ImageNet/resnet50/cSGD/seed1/PCA/metrics_pca.csv') %>%
      mutate(seed='1')
    ) %>%
  union(
    read.csv('lgv/results/ImageNet/resnet50/cSGD/seed2/PCA/metrics_pca.csv') %>%
      mutate(seed='2')
  ) %>%
  mutate(
    n_dims = dim+1  # dim=index of dims, n_dims=nb of dims
  ) %>%
  select(seed, n_dims, totcum_expl_var)

df <- df %>%
  left_join(df_pca, by=c('seed', 'n_dims')) %>%
  mutate(
    totcum_expl_var = case_when(
      n_dims==0 ~ 0.,
      n_dims==40 ~ 1.,
      TRUE ~ totcum_expl_var
    )
  )

# check that we have 3 seeds per group
df %>%
  # group_by(Norm, Dataset, n_dims) %>%
  group_by(Norm, Dataset, `Target`, n_dims) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate),
    mean(transfer_rate),
    mean_totcum_expl_var=mean(totcum_expl_var),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  ) %>% 
  mutate(
    diff_success_rate = mean_adv_success_rate - lag(mean_adv_success_rate),
    diff_expl_var = mean_totcum_expl_var - lag(mean_totcum_expl_var)
    ) %>% View()


# baseline: equal contributions of all directions baseline
#add "theoretical" lines connecting SWA and collected models if weights varaince don't matter, and success rate would be spited equally between directions
df_baseline <- df_pca %>% 
  group_by(n_dims) %>%
  summarise(mean_totcum_expl_var = mean(totcum_expl_var)) %>%
  ungroup() %>%
  rows_insert(tibble(n_dims = c(0,40), mean_totcum_expl_var = c(0., 1.))) %>%
  crossing(df %>% expand(Norm, Dataset, `Target`)) %>%
  left_join(
    df %>%
      filter(n_dims == 0) %>%
      group_by(Norm, Dataset, `Target`) %>%
      summarise(
        mean_adv_success_rate_swa = mean(adv_success_rate),
      )
  ) %>% left_join(
    df %>%
      filter(n_dims == 40) %>%
      group_by(Norm, Dataset, `Target`) %>%
      summarise(
        mean_adv_success_rate_full = mean(adv_success_rate),
      )
  ) %>%
  transmute(
    Norm, Dataset, `Target`,
    n_dims, mean_totcum_expl_var,
    mean_adv_success_rate_baseline = mean_adv_success_rate_swa+(mean_adv_success_rate_full - mean_adv_success_rate_swa)/40*n_dims,
    mean_adv_success_rate_linearity = mean_adv_success_rate_swa+(mean_adv_success_rate_full - mean_adv_success_rate_swa)*mean_totcum_expl_var
  )


ggplot(df, aes(x = totcum_expl_var, y = adv_success_rate, shape = Norm, colour=Norm, fill=Norm, group = interaction(Norm))) + # linetype=Norm
  # loess: smooth lines with SE
  {if(!main_plot) geom_smooth(method = "loess", stat = "smooth", se = TRUE, alpha=0.2, span=0.4) } + # span=0.4: 40% points
  # lm: linear relation
  geom_point(alpha=0.5, size=3) +
  geom_point(size=0.1, colour='black') +
  xlab("Explained Weights Variance Ratio") +
  ylab("Success Rate") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  {if(main_plot) scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits=c((main_plot_arch=='ResNet-50')*0.83+(main_plot_arch=='Inception v3')*0.106, NA))} +
  {if(!main_plot) scale_y_continuous(labels = scales::percent_format(accuracy = 1))} +
  {if(main_plot & main_plot_arch=='ResNet-50') geom_line(data = df_baseline, aes(x = mean_totcum_expl_var, y = mean_adv_success_rate_baseline, colour=Norm, group=Norm), alpha=0.3, linetype='longdash', inherit.aes=FALSE, size=1.)} +
  {if(main_plot & main_plot_arch=='ResNet-50') geom_line(data = df_baseline, aes(x = mean_totcum_expl_var, y = mean_adv_success_rate_linearity, colour=Norm, group=Norm), alpha=0.9, inherit.aes=FALSE, size=1.)} +
  {if(main_plot & main_plot_arch=='Inception v3') geom_line(data = df_baseline, aes(x = mean_totcum_expl_var, y = mean_adv_success_rate_baseline, colour=Norm, group=Norm), alpha=0.9, linetype='longdash', inherit.aes=FALSE, size=1.)} +
  {if(main_plot & main_plot_arch=='Inception v3') geom_line(data = df_baseline, aes(x = mean_totcum_expl_var, y = mean_adv_success_rate_linearity, colour=Norm, group=Norm), alpha=0.3, inherit.aes=FALSE, size=1.)} +
  {if(!main_plot) geom_line(data = df_baseline, aes(x = mean_totcum_expl_var, y = mean_adv_success_rate_baseline, colour=Norm, group=Norm), alpha=0.3, linetype='longdash', inherit.aes=FALSE, size=1.)} +
  {if(main_plot & main_plot_arch=='ResNet-50') geom_label(x=0.04, y=0.827, label="LGV-SWA", size=3.88, label.padding=unit(0.17, "lines"), inherit.aes = FALSE)} + # label.padding=unit(0.15, "lines")
  {if(main_plot & main_plot_arch=='ResNet-50') geom_label(x=1., y=0.827, label="LGV", size=3.88, hjust='center', vjust='center', label.padding=unit(0.17, "lines"), inherit.aes = FALSE)} +
  {if(main_plot & main_plot_arch=='Inception v3') geom_label(x=0.04, y=0.1, label="LGV-SWA", size=3.88, label.padding=unit(0.17, "lines"), inherit.aes = FALSE)} + # label.padding=unit(0.15, "lines")
  {if(main_plot & main_plot_arch=='Inception v3') geom_label(x=1., y=0.1, label="LGV", size=3.88, hjust='center', vjust='center', label.padding=unit(0.17, "lines"), inherit.aes = FALSE)} +
  {if(!main_plot) facet_wrap( ~ `Target`, nrow=2, scales="free_y")}
filename_export <- if_else(main_plot, paste0('lgv/plots/rq1_proj_dims_lm_', str_replace_all(main_plot_arch, ' ', '_'),'.pdf'),'lgv/plots/rq1_proj_dims_loess_interarch.pdf')
ggsave(filename=filename_export, width=17.43*(2-main_plot), height=13.79, units="cm", device=cairo_pdf)  # size of template example figure
if(main_plot) ggsave(filename=str_replace(filename_export, '.pdf', '.png'), width=17.43*(2-main_plot), height=13.79, units="cm")  # size of template example figure

# check individual lines
ggplot(df, aes(x = totcum_expl_var, y = adv_success_rate, colour=Norm, group = Norm)) + 
  geom_point(alpha=0.8) + 
  geom_line(aes(color=Norm, group=id, linetype=id)) +
  xlab("Explained Weights Variance Ratio") +
  ylab("Success Rate") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) 


##############################
#  RQ1 - GAUSSIAN IN SUBSPACE
##############################

filename <- 'lgv/results/ImageNet/resnet50/RQ1/attack_gaussian_subspace_interarch.csv'
df = read.csv(filename) %>%
  mutate(
    Surrogate_ = case_when(
      grepl('noisy/gaussian_subspace', model_surrogate) ~ 'LGV-SWA + RD in LGV Subspace',
      grepl('/noisy/random_ensemble_equivalent', model_surrogate) ~ 'LGV-SWA + RD (equiv)',
      grepl('/PCA/dims_0/noisy/std_0.01_50models', model_surrogate) ~ 'LGV-SWA + RD',
      grepl('PCA/dims_0', model_surrogate) ~ 'LGV-SWA',
      grepl('/seed[0-9]$', model_surrogate) ~ 'LGV',
      TRUE ~ NA_character_
    ),
    Surrogate = factor(Surrogate_, levels=c('LGV', 'LGV-SWA + RD in LGV Subspace', 'LGV-SWA + RD',  'LGV-SWA + RD (equiv)', 'LGV-SWA')),
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    Target = recode_archs(arch_target, abbr = T)
  ) %>%
  # remove 50 models sampled from the subspace
  filter(!(Surrogate == 'LGV-SWA + RD in LGV Subspace' & surrogate_size_ensembles < 50), Surrogate != 'LGV-SWA')

# check that we have 3 seeds
df_ <- df %>%
  group_by(Norm, `Target`, Surrogate) %>%
  filter(Surrogate %in% c('LGV', 'LGV-SWA + RD in LGV Subspace', 'LGV-SWA + RD', 'LGV-SWA')) %>%
  # LGV-SWA+RD run twice
  filter(!(Surrogate=='LGV-SWA + RD' & row_number() %in% c(1,2,3))) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate * 100),
    sd_adv_success_rate = sd(adv_success_rate * 100),
    mean(transfer_rate),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  )
View(df_)

df__ <- df_ %>%
  transmute(
    Target,
    Surrogate,
    success_rate_chr = paste0(formatC(mean_adv_success_rate, digits=1, format='f'), 'SMALL ±', formatC(sd_adv_success_rate, digits=1, format='f'))
  ) %>%
  pivot_wider(names_from = `Target`, values_from = success_rate_chr) %>%
  ungroup()

View(df__)

debug_table <- F  # if T, print HTML, otherwise output LaTeX code
caption_ <- "Transfer success rate of random directions sampled in LGV deviations subspace."
table <- kbl(df__, booktabs = T, caption = caption_, format = ifelse(debug_table, 'html', 'latex'), align = c('l', 'l', rep("r", n_targets)), linesep = "") %>%
  kable_styling(latex_options = c("hold_position", 'striped'), font_size=9) %>% # "scale_down" 
  add_header_above(c(" ", " ", "Target" = n_targets)) %>%
  column_spec(2, width = '5em')
table

patch_table <- function(str) {
  str <- gsub('∞', "$\\infty$", str, fixed=T)
  str <- sub('\\begin{table}[!h]', '\\begin{table}[!ht]', str, fixed=T)
  str <- sub('\\end{table}', '\n\\label{tab:tgv_gaussian_subspace} \n \\end{table}', str, fixed=T)
  str <- gsub('SMALL', '\\tiny', str, fixed=T)
  str <- gsub('LGV-SWA + RD in LGV Subspace', 'LGV-SWA + RD in $\\mathcal{S}$', str, fixed=T)
  return(str)
}

fileConn<-file("lgv/plots/table_tgv_gaussian_subspace.tex")
writeLines(patch_table(table), fileConn)
close(fileConn)

# analyse
df_ %>% 
  ungroup() %>%
  left_join(
    df_ %>%
      filter(Surrogate == 'LGV-SWA + RD in LGV Subspace') %>%
      transmute(mean_adv_success_rate_tgv_rd_subspace = mean_adv_success_rate, Norm, `Target`)
  ) %>%
  mutate(
    diff_success_rate_tgv_rd_subspace = mean_adv_success_rate - mean_adv_success_rate_tgv_rd_subspace,
  ) %>%
  # mean improvements
  #group_by(Surrogate) %>% summarise(mean(diff_success_rate_tgv_rd_subspace))
  View()


#######################################################
#  RQ1 - DEVIATIONS FROM ANOTHER LGV LOCAL MAXIMA
#######################################################

df <- read_csv('lgv/results/ImageNet/resnet50/RQ1/attack_tgv_swa_translated_interarch.csv', show_col_types = FALSE) %>%
  mutate(
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Target = recode_archs(arch_target, abbr = TRUE),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    Surrogate_ = case_when(
      grepl('original/translation/translation_deviations_to_new_swa', model_surrogate) ~ '1 DNN + (LGV\' - LGV-SWA\')',
      grepl('translation_indiv/translation_deviations_to_new_swa', model_surrogate) ~ 'LGV indiv. + (LGV\' - LGV-SWA\')',
      grepl('translation/translation_deviations_to_new_swa', model_surrogate) ~ 'LGV-SWA + (LGV\' - LGV-SWA\')',
      grepl('/original$', model_surrogate) ~ '1 DNN',
      grepl('/noisy/random_ensemble_equivalent', model_surrogate) ~ 'LGV-SWA + RD (equiv)',
      grepl('/PCA/dims_0/noisy/std_0.01_50models', model_surrogate) ~ 'LGV-SWA + RD',
      grepl('/original/noisy/std_0.005_50models', model_surrogate) ~ '1 DNN + RD',
      grepl('/PCA/dims_0', model_surrogate) ~ 'LGV-SWA',
      grepl('/cSGD/seed[0-2]$', model_surrogate) ~ 'LGV (ours)',
      grepl('/cSGD/seed[3-9]$', model_surrogate) ~ 'LGV\' (ours)',
      TRUE ~ NA_character_
    ),
    Surrogate = factor(Surrogate_, levels = c('LGV-SWA + (LGV\' - LGV-SWA\')', 'LGV-SWA + RD', 'LGV (ours)', '1 DNN + (LGV\' - LGV-SWA\')', '1 DNN + RD'))
  ) %>%
  filter(!is.na(Surrogate))

df_ <- df %>%
  group_by(Norm, `Target`, Surrogate) %>%
  summarise(
    mean_adv_success_rate = mean(adv_success_rate * 100),
    sd_adv_success_rate = sd(adv_success_rate * 100),
    mean(transfer_rate* 100),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate),
  )
View(df_)
if(df_ %>% filter(n_seeds != 3) %>% nrow()) warning('Wrong number of seeds!')


archs_resnet <- T # True for only resnet like targets, F other targets
archs_resnet <- F

if (archs_resnet) df__ <- df_ %>% filter(Target %in% c("RN50", "RN152", "RNX50", "WRN50"))
if (!archs_resnet) df__ <- df_ %>% filter(Target %in% c("DN201", "VGG19", "IncV1", "IncV3"))
n_targets_plot <- 4

df__ <- df__ %>%
  filter(Surrogate != 'LGV\' (ours)') %>% # remove unreported baseline
  transmute(
    `Target`,
    Surrogate,
    success_rate_chr = paste0(formatC(mean_adv_success_rate, digits=1, format='f'), 'SMALL ±', formatC(sd_adv_success_rate, digits=1, format='f'))
  ) %>%
  pivot_wider(names_from = `Target`, values_from = success_rate_chr) %>%
  ungroup()

View(df__)

debug_table <- F  # if T, print HTML, otherwise output LaTeX code
caption_ <- if_else(archs_resnet, 
                    "Transfer success rate of LGV deviations shifted to other independent solutions, for target architectures in the ResNet family.",
                    "Transfer success rate of LGV deviations shifted to other independent solutions, for non-ResNet targets.")
table <- kbl(df__, booktabs = T, caption = caption_, format = ifelse(debug_table, 'html', 'latex'), align = c('l', 'l', rep("r", n_targets_plot)), linesep = "") %>%
  kable_styling(latex_options = c("hold_position", 'striped'), font_size=9) %>% # "scale_down" 
  add_header_above(c(" ", " ", "Target" = n_targets_plot))# %>%
  # column_spec(2, width = '5em')
table

patch_table <- function(str) {
  str <- gsub('∞', "$\\infty$", str, fixed=T)
  str <- sub('\\end{table}', paste0('\n\\label{tab:tgv_swa_translated', ifelse(archs_resnet, '_resnet_target', '_other_target'), '} \n \\end{table}'), str, fixed=T)
  str <- gsub('SMALL', '\\tiny', str, fixed=T)
  str <- gsub('1 DNN + (LGV', '1 DNN + $\\gamma$ (LGV', str, fixed=T)
  # str <- gsub(' + ', '+', str, fixed=T)
  # str <- gsub(' - ', '-', str, fixed=T)
  return(str)
}

fileConn<-file(if_else(archs_resnet, "lgv/plots/table_tgv_swa_translated_resnet_target.tex", "lgv/plots/table_tgv_swa_translated_other_target.tex"))
writeLines(patch_table(table), fileConn)
close(fileConn)

# compute the difference with several baselines
df_ %>% 
  ungroup() %>%
  left_join(
    df_ %>%
      filter(Surrogate == 'LGV (ours)') %>%
      transmute(mean_adv_success_rate_tgv = mean_adv_success_rate, Norm, `Target`)
  ) %>%
  left_join(
    df_ %>%
      filter(Surrogate == 'LGV-SWA + RD') %>%
      transmute(mean_adv_success_rate_tgv_swa_rd = mean_adv_success_rate, Norm, `Target`)
  ) %>%
  left_join(
    df_ %>%
      filter(Surrogate == 'LGV-SWA + (LGV\' - LGV-SWA\')') %>%
      transmute(mean_adv_success_rate_tgv_swa_translated = mean_adv_success_rate, Norm, `Target`)
  ) %>%
  left_join(
    df_ %>%
      filter(Surrogate == '1 DNN + (LGV\' - LGV-SWA\')') %>%
      transmute(mean_adv_success_rate_dnn_swa_translated = mean_adv_success_rate, Norm, `Target`)
  ) %>%
  mutate(
    diff_success_rate_tgv = mean_adv_success_rate_tgv - mean_adv_success_rate,
    diff_success_rate_tgv_swa_rd = mean_adv_success_rate_tgv_swa_rd - mean_adv_success_rate,
    diff_success_rate_tgv_swa_translated = mean_adv_success_rate_tgv_swa_translated - mean_adv_success_rate,
    diff_success_rate_dnn_swa_translated = mean_adv_success_rate_dnn_swa_translated - mean_adv_success_rate,
  ) %>%
  # mean improvements
  #filter(Surrogate == 'LGV-SWA + (LGV\' - LGV-SWA\')') %>% summarise(mean(diff_success_rate_tgv), mean(diff_success_rate_tgv_swa_rd))
  #filter(Surrogate == '1 DNN + RD') %>% summarise(mean(diff_success_rate_dnn_swa_translated))
  # group_by(Surrogate) %>% summarise(mean(diff_success_rate_tgv_swa_translated))
  View()



##################
#  HP - NB ITERS  
##################


df = read.csv('lgv/results/ImageNet/resnet50/cSGD/HP/iters/attack_interarch.csv') %>%
  mutate(
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    Surrogate = if_else(grepl('original', model_surrogate), '1 DNN', 'LGV'),
    `Target` = recode_archs(arch_target),
  ) %>%
  unite(id, c("Norm", "Dataset", "seed"), remove=FALSE)

df %>%
  group_by(Norm, Dataset, `Target`, Surrogate, n_iter) %>%
  summarise(
    mean(adv_success_rate),
    sd(adv_success_rate),
    mean(transfer_rate),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  ) %>% View()

ggplot(df, aes(x = n_iter, y = adv_success_rate, shape = Dataset, linetype=Dataset, colour=Norm, fill=Norm, group = interaction(Norm, Dataset))) + 
  stat_summary(fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # mean +/- 1 std
  xlab("Number of Iterations") +
  ylab("Success Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  # theme_light() +
  facet_grid(Surrogate ~ `Target`) +
  theme(strip.text.x = element_text(size=13))
ggsave(filename='lgv/plots/hp_nb_iters_interarch.pdf', width=17.43*2, height=13.79, units="cm", device=cairo_pdf)



####################
#  HP - NB EPOCHS  #
####################

df = read.csv('lgv/results/ImageNet/resnet50/cSGD/HP/epochs/attack_interarch.csv') %>%
  mutate(
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target),
    Epochs = as.integer(replace_na(limit_cycles, 0) / 4)  # 0 is the original model
  ) %>%
  unite(id, c("Norm", "Dataset", "seed"), remove=FALSE)

df %>%
  group_by(Norm, Dataset, `Target`, Epochs) %>%
  summarise(
    mean(adv_success_rate),
    sd(adv_success_rate),
    mean(transfer_rate),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  ) %>% View()

ggplot(df, aes(x = Epochs, y = adv_success_rate, shape = Dataset, linetype=Dataset, colour=Norm, fill=Norm, group = interaction(Norm, Dataset))) + 
  stat_summary(fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # mean +/- 1 std
  xlab("Number of Epochs") +
  ylab("Success Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_x_continuous(breaks = integer_breaks()) +
  facet_wrap( ~ `Target`, nrow=2)
ggsave(filename='lgv/plots/hp_nb_epochs_interarch.pdf', width=17.43*2, height=13.79, units="cm", device=cairo_pdf)



##############################
#  HP - NB MODELS PER EPOCH  #
##############################

df = read.csv('lgv/results/ImageNet/resnet50/cSGD/HP/nb_models/attack_interarch.csv') %>%
  mutate(
    seed = str_match(model_surrogate, "seed(\\d+)")[,2],
    Dataset = recode(dataset, "val" = "Val.", "test" = "Test"),
    Norm = recode(norm_type, "2" = "L2", "Inf" = "L∞"),
    `Target` = recode_archs(arch_target),
    nb_models = as.integer(replace_na(limit_samples_cycle, 0))  # 0 is the original model
  ) %>%
  unite(id, c("Norm", "Dataset", "seed"), remove=FALSE)

df %>%
  group_by(Norm, Dataset, `Target`, nb_models) %>%
  summarise(
    mean(adv_success_rate),
    sd(adv_success_rate),
    mean(transfer_rate),
    n_seeds = n(),
    nb_ex = mean(nb_adv),
    nb_ex_transfRate = mean(nb_adv_transf_rate)
  ) %>% View()

ggplot(df, aes(x = nb_models, y = adv_success_rate, shape = Dataset, linetype=Dataset, colour=Norm, fill=Norm, group = interaction(Norm, Dataset))) + 
  stat_summary(fun.data = "mean_se", geom = "smooth", se = TRUE, alpha=0.4) +  # mean +/- 1 std
  xlab("Number of Models per Epoch") +
  ylab("Success Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_x_continuous(breaks = integer_breaks()) +
  facet_wrap( ~ `Target`, nrow=2)
ggsave(filename='lgv/plots/hp_nb_models_interarch.pdf', width=17.43*2, height=13.79, units="cm", device=cairo_pdf)



############################
#  DIAGRAM - Learning Rate 
############################

df <- data.frame(
  epochs = c(0,30, 30,60, 60,90, 90,120, 120,130, 130,130+10),
  lr = c(0.1,0.1, 0.01,0.01, 0.001,0.001, 0.0001,0.0001, 0.00001,0.00001, 0.05,0.05)
)

df_samples <- data.frame(
  epochs=c((130:139)+0.25, (130:139)+0.5, (130:139)+0.75,131:140),
  lr=0.05
) %>%
  arrange(epochs) %>%
  mutate(color=0:39)

ggplot(df, aes(x=epochs, y=lr)) +
  geom_step(color='#977ac2', size=1) +
  geom_point(aes(colour=color), data=df_samples, alpha=0.7, show.legend=FALSE, size=2.6, shape=21, stroke=1.3) +
  scale_colour_gradientn(colours = c('#66FFFF', '#3399FF', '#000099')) +
  geom_curve(
    aes(x = 110, y = 0.045, xend = 130-2.5, yend = 0.05),
    arrow = arrow(
      length = unit(0.04, "npc"), 
    ),
    alpha=0.2,
    colour = "#3399FF",
    size = 0.7,
    #angle = 0, # Anything other than 90 or 0 can look unusual
    curvature=0.2,
    inherit.aes = FALSE
  ) +
  geom_label(x=105, y=0.047, label="LGV\nSurrogate", size=5, hjust='center', vjust='center', inherit.aes = FALSE, colour="#3399FF") + 
  scale_x_continuous(breaks=unique(df$epochs)) +
  geom_segment(
    x = 130, y = 0.075-0.007,
    xend = 140, yend = 0.075-0.007,
    lineend = "round",
    linejoin = "round",
    size = 1,
    arrow = arrow(ends='both', length = unit(0.12, "inches")),
    colour = "#EC7014"
  ) +
  geom_label(x=135-1, y=0.075+0.011, label="Weights\nCollection", size=5, hjust='center', vjust='center', inherit.aes = FALSE) + 
  geom_segment(
    x = 1, y = 0.075-0.007,
    xend = 130-2, yend = 0.075-0.007,
    lineend = "round",
    linejoin = "round",
    size = 1,
    arrow = arrow(ends='both', length = unit(0.12, "inches")),
    colour = "#EC7014"
  ) +
  geom_label(x=130/2, y=0.075+0.011, label="Regular\nTraining", size=5, hjust='center', vjust='center', inherit.aes = FALSE) + 
  geom_point(x=130, y=0.00001, alpha=0.9, size=3, inherit.aes = FALSE, color='gray33', fill='white', shape=21, stroke=1.3) +
  geom_curve(
    aes(x = 110, y = 0.025/2, xend = 130-1.8, yend = 0.0025),
    arrow = arrow(
      length = unit(0.04, "npc"), 
    ),
    alpha=0.2,
    colour = "gray33",
    size = 0.7,
    #angle = 0, # Anything other than 90 or 0 can look unusual
    curvature=-0.2,
    inherit.aes = FALSE
  ) +
  geom_label(x=110, y=0.025/2+0.003, label="Initial\nDNN", size=5, hjust='center', vjust='center', inherit.aes = FALSE) + 
  xlab("Epoch") +
  ylab("Learning Rate") +
  ylim(0, 0.103) +
  facet_zoom(xlim = c(130, 140), zoom.size=2/3)
ggsave(filename='lgv/plots/diagram_lr.pdf', width=17.43, height=13.79, units="cm", device=cairo_pdf)
ggsave(filename='lgv/plots/diagram_lr.png', width=17.43, height=13.79, units="cm")
