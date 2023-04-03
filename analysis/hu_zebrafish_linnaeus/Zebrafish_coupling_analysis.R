# Dependencies ####
library(data.table)
library(ggplot2)
library(plotROC)
library(ggalluvial)

setwd("./Scripts/notebooks/hu_zebrafish_linnaeus/")
tree_time <- fread("Tree_times")

AverageFrequenciesAndTransferRatios <- function(transfer_ratios, tree_times, all_ct_combinations = NULL){
  # Perform a weighted average of cell type frequencies and transfer ratios.
  # Cell type frequencies are averaged over datasets per timepoint and weighted by the size of each dataset.
  # Transfer ratios are averaged over dataset combinations per timepoint combination and weighted by
  # the product of the sizes (divided by total_t1*total_t2, with total_t1 := sum(#t1)).
  # Average frequencies and transfer ratios over trees.
  moslin_ct_freqs_from <- 
    unique(transfer_ratios[, c("Cell_type_from", "Tree_from", "ct_from_freq", "ct_from_rel_freq")])
  colnames(moslin_ct_freqs_from) <- c("Cell_type", "Tree", "Freq", "Rel_freq")
  moslin_ct_freqs_to <- 
    unique(transfer_ratios[, c("Cell_type_to", "Tree_to", "ct_to_freq", "ct_to_rel_freq")])
  colnames(moslin_ct_freqs_to) <- c("Cell_type", "Tree", "Freq", "Rel_freq")
  moslin_ct_freqs <- unique(rbind(moslin_ct_freqs_from, moslin_ct_freqs_to))
  moslin_ct_freqs[, Tree_size := sum(Freq), by = "Tree"]
  moslin_ct_freqs <- merge(moslin_ct_freqs, tree_times, by = "Tree")
  # Do relative frequencies sum to 1 per tree?
  # moslin_ct_freqs[, sum(Rel_freq), by = "Tree"] # should all be 1
  
  all_ct_freqs <- data.table(expand.grid(Tree = unique(moslin_ct_freqs$Tree),
                                         Cell_type = unique(moslin_ct_freqs$Cell_type)))
  # Merge in two steps to ensure tree size and time are filled for all records in all_ct_freqs
  all_ct_freqs <- 
    merge(all_ct_freqs, unique(moslin_ct_freqs[, c("Tree", "Tree_size", "time")]),
          by = "Tree")
  all_ct_freqs <- 
    merge(all_ct_freqs, 
          moslin_ct_freqs[, c("Tree", "Cell_type", "Rel_freq")],
          by = c("Tree", "Cell_type"), all = T)
  all_ct_freqs$Rel_freq[is.na(all_ct_freqs$Rel_freq)] <- 0
  mean_ct_freqs <- 
    all_ct_freqs[, .(Rel_freq = weighted.mean(Rel_freq, Tree_size)), 
                 by = c("Cell_type", "time")]
  # Do average relative frequencies sum to 1 per timepoint?
  # mean_ct_freqs[, sum(Rel_freq), by = "time"]
  
  # transfer_ratios[, sum(Transfer_ratio), by = c("Tree_from", "Tree_to")]
  if(is.null(all_ct_combinations)){
    ct_tofrom <- unique(paste(transfer_ratios$Cell_type_from, transfer_ratios$Cell_type_to, sep = "_"))
    trees_tofrom <- unique(paste(transfer_ratios$Tree_from, transfer_ratios$Tree_to, sep = "_"))
    all_ct_combinations <-
      data.table(expand.grid(ct_tf = ct_tofrom,
                             trees_tf = trees_tofrom, stringsAsFactors = F))
    all_ct_combinations <-
      all_ct_combinations[, {
        cell_types <- unlist(strsplit(ct_tf, "_"))
        trees <- unlist(strsplit(trees_tf, "_"))
        list(Cell_type_from = cell_types[1],
             Cell_type_to = cell_types[2],
             Tree_from = trees[1],
             Tree_to = trees[2])
      }, by = names(all_ct_combinations)]
  }
  transfer_ratios <- 
    merge(transfer_ratios, all_ct_combinations[, c("Cell_type_from", "Cell_type_to", "Tree_from", "Tree_to")], 
          by = c("Cell_type_from", "Cell_type_to", "Tree_from", "Tree_to"),
          all = T)
  transfer_ratios$Transfer_ratio[is.na(transfer_ratios$Transfer_ratio)] <- 0
  # transfer_ratios[, sum(Transfer_ratio), by = c("Tree_from", "Tree_to")]
  # Add tree sizes (i.e. number of cells) for t_1 and t_2 tree.
  transfer_ratios <- merge(transfer_ratios, unique(moslin_ct_freqs[, c("Tree", "Tree_size", "time")]),
                           by.x = "Tree_from", by.y = "Tree")
  colnames(transfer_ratios)[c(ncol(transfer_ratios) - 1, ncol(transfer_ratios))] <-
    c("t1_size", "t1_time")
  transfer_ratios <- merge(transfer_ratios, unique(moslin_ct_freqs[, c("Tree", "Tree_size", "time")]),
                           by.x = "Tree_to", by.y = "Tree")
  colnames(transfer_ratios)[c(ncol(transfer_ratios) - 1, ncol(transfer_ratios))] <-
    c("t2_size", "t2_time")
  # Calculate total amount of cells at different timepoints.
  time_sizes <- unique(all_ct_freqs[, c("Tree", "Tree_size", "time")])[, .(Time_total = sum(Tree_size)), by = "time"]
  transfer_ratios <-
    merge(transfer_ratios, time_sizes, by.x = "t1_time", by.y = "time")
  colnames(transfer_ratios)[ncol(transfer_ratios)] <- "t1_total"
  transfer_ratios <-
    merge(transfer_ratios, time_sizes, by.x = "t2_time", by.y = "time")
  colnames(transfer_ratios)[ncol(transfer_ratios)] <- "t2_total"
  # Perform a weighted average using weights #t1 * #t2/(#Total_t1 * # Total_t2).
  transfer_ratios <- 
    transfer_ratios[, Weight_factor := t1_size * t2_size/(t1_total * t2_total)]
  mean_transfer_ratios <- 
    transfer_ratios[, .(Transfer_ratio = sum(Weight_factor * Transfer_ratio)),
                    by = c("t1_time", "t2_time", "Cell_type_from", "Cell_type_to")]
  # mean_transfer_ratios[, sum(Transfer_ratio), by = "t1_time"] # should be 1
  # View(mean_transfer_ratios[t2_time == "3dpi", sum(Transfer_ratio), by = c("Cell_type_to")])
  # View(mean_ct_freqs[time == "3dpi"])
  # # should be equal; is almost equal and I think this has to do with normalization being not entirely 1
  # View(mean_transfer_ratios[t2_time == "7dpi", sum(Transfer_ratio), by = c("Cell_type_to")])
  # View(mean_ct_freqs[time == "7dpi"])
  # # should be equal; is almost equal and I think this has to do with normalization being not entirely 1
  
  return(list(Cell_type_frequencies = mean_ct_freqs, Transfer_ratios = mean_transfer_ratios))
}

CreateAlluvialDT <- function(transfer_averages, ct_averages){
  # Create an alluvial dataframe from cell type (frequency) averages and transfer (ratio) averages.
  
  t1t2_transfers <- transfer_averages[t1_time == "Ctrl"]
  colnames(t1t2_transfers)[c(3:5)] <- c("Type_t1", "Type_t2", "Transfer_t2")
  t1t2_alluvia <-
    merge(t1t2_transfers, ct_averages, 
          by.x = c("Type_t1", "t1_time"), by.y = c("Cell_type", "time"))
  colnames(t1t2_alluvia)[ncol(t1t2_alluvia)] <- "Rel_freq_type_t1"
  t1t2_alluvia <- t1t2_alluvia[, Mass_transfer_from_type_t1 := sum(Transfer_t2), by = "Type_t1"]
  t1t2_alluvia$Transfer_t1 <- t1t2_alluvia$Transfer_t2 * t1t2_alluvia$Rel_freq_type_t1/t1t2_alluvia$Mass_transfer_from_type_t1
  # Remove NaN's caused by division by zero
  t1t2_alluvia$Transfer_t1[t1t2_alluvia$Transfer_t2 == 0] <- 0
  # View(t1t2_alluvia[, sum(Transfer_t1), by = "Type_t1"])
  # View(ct_averages[time == "Ctrl"])
  # # Should be equal
  # t1t2_alluvia$Alluvium <- paste(t1t2_alluvia$Type_t1, t1t2_alluvia$Type_t2, sep = "_")
  
  t2t3_transfers <- transfer_averages[t1_time == "3dpi", c("Cell_type_from", "Cell_type_to", "Transfer_ratio")]
  colnames(t2t3_transfers)[1:2] <- c("Type_t2", "Type_t3")
  t1t2t3_alluvia <- 
    merge(t1t2_alluvia[, c("Type_t1", "Type_t2", "Transfer_t1", "Transfer_t2")],  #"Alluvium", 
          t2t3_transfers, by.x = "Type_t2", by.y = "Type_t2", allow.cartesian = T, all = T)
  colnames(t1t2t3_alluvia)[c(3:4, ncol(t1t2t3_alluvia))] <- 
    c("t1t2_alluvium_size_t1", "t1t2_alluvium_size_t2", "Transfer_t2t3") #"t1t2_alluvium", 
  t1t2t3_alluvia$Type_t1[is.na(t1t2t3_alluvia$Type_t1)] <- "Notshow"
  t1t2t3_alluvia$t1t2_alluvium <- paste(t1t2t3_alluvia$Type_t1, t1t2t3_alluvia$Type_t2, sep = "_")
  t1t2t3_alluvia$t1t2_alluvium_size_t1[is.na(t1t2t3_alluvia$t1t2_alluvium_size_t1)] <- 0
  t1t2t3_alluvia$t1t2_alluvium_size_t2[is.na(t1t2t3_alluvia$t1t2_alluvium_size_t2)] <- 0
  # To determine the sizes of the t123 alluvia, start by subdividing the t1t2 alluvia sizes by
  # the t3 ratios. First calculate the t3 ratios:
  t1t2t3_alluvia <- t1t2t3_alluvia[, t1t2_alluvium_ratio_t3 := sum(Transfer_t2t3), by = "t1t2_alluvium"]
  # Subdivide the t1 size of the t1t2 alluvia
  t1t2t3_alluvia$t1t2t3_alluvium_size_t1 <-
    t1t2t3_alluvia$t1t2_alluvium_size_t1 * t1t2t3_alluvia$Transfer_t2t3/t1t2t3_alluvia$t1t2_alluvium_ratio_t3
  # View(t1t2t3_alluvia[, .(Transfer_t1 = sum(t1t2t3_alluvium_size_t1)), by = "t1t2_alluvium"])
  # View(t1t2_alluvia[, c("Alluvium", "Transfer_t1")])
  # # Should be the same
  # Subdivide the t2 size of the t1t2 alluvia
  t1t2t3_alluvia$t1t2t3_alluvium_size_t2 <-
    t1t2t3_alluvia$t1t2_alluvium_size_t2 * t1t2t3_alluvia$Transfer_t2t3/t1t2t3_alluvia$t1t2_alluvium_ratio_t3
  # View(t1t2t3_alluvia[, .(Transfer_t2 = sum(t1t2t3_alluvium_size_t2)), by = "t1t2_alluvium"])
  # View(t1t2_alluvia[, c("Alluvium", "Transfer_t2")])
  # # Should be the same
  # At t3, the size of the t1t2t3 alluvium should be the size of Transfer_t2t3, divided pro rata
  # of the t2 sizes of the t1t2 alluvia that have the same t2 cell type.
  t1t2t3_alluvia <- t1t2t3_alluvia[, c("t2t3_size_t2", "ct1_counter") :=
                                     list(sum(t1t2_alluvium_size_t2), length(Type_t1)),
                                   by = c("Type_t2", "Type_t3")]
  t1t2t3_alluvia <- 
    t1t2t3_alluvia[, 
                   Proportion_t3 := ifelse(ct1_counter == 1, 1,
                                           t1t2_alluvium_size_t2/t2t3_size_t2),
                   by = names(t1t2t3_alluvia)]
  t1t2t3_alluvia$t1t2t3_alluvium_size_t3 <-
    t1t2t3_alluvia$Proportion_t3 * t1t2t3_alluvia$Transfer_t2t3
  # # This should sum to Transfer_t2t3:
  # View(t1t2t3_alluvia[, .(Transfer_t2t3 = sum(t1t2t3_alluvium_size_t3)), by = c("Type_t2", "Type_t3")])
  # View(t2t3_transfers)
  # # And to t3 cell type frequencies:
  # View(t1t2t3_alluvia[, .(Rel_freq = sum(t1t2t3_alluvium_size_t3)), by = "Type_t3"])
  # View(ct_averages[time == "7dpi", c("Cell_type", "Rel_freq")])
  t1t2t3_alluvia$Alluvium <- paste(t1t2t3_alluvia$t1t2_alluvium, t1t2t3_alluvia$Type_t3, sep = "_")
  
  return(t1t2t3_alluvia)
}

CreateAlluvialPlotDT <- function(alluvia_dt, ct_averages, ct_relative = F, fraction_threshold = 0.001){
  # Important part of this is the determination which connections should be plotted
  # and which ones should not.
  # Which connections do we want to plot? The ones that are large with respect to
  # the source or target frequencies. Good baseline here is as in Qiu et al., to
  # see which ones are same-type propagating and what their sizes are.
  # In t1t2t3_alluvia, the relevant columns for flows are t1t2_alluvium_size_t1, t1t2_alluvium_size_t2,
  # t2t3_size_t2 and Transfer_t2t3. Add cell type frequencies to this to compare, determine and set cutoff.
  # Flows are removed by setting alpha -> 0 for the left side of the flow; i.e. a ctrl-3dpi flow is removed
  # by setting alpha = 0 on its ctrl value. Not every flow of an alluvium needs to be shown, i.e. we can
  # decide which flow to show and which to not show. In particular, this means we should check ctrl-3dpi flows
  # for relative left/right size < 0.01 and 3dpi-7dpi flows for relative left/right size < 0.01 separately.
  # We can add two separate columns to determine the alpha at t1 and at t2 and transfer those to the
  # alluvium plot dataframe.
  
  # For ct_relative == T, remove alluvia based on size relative to cell type frequency.
  if(ct_relative){
    alluvia_dt <- 
      merge(alluvia_dt,  
            ct_averages[time == "7dpi", c("Cell_type", "Rel_freq")], 
            by.x = "Type_t3", by.y = "Cell_type")
    colnames(alluvia_dt)[ncol(alluvia_dt)] <- "Rel_freq_t3"
    alluvia_dt <-
      merge(alluvia_dt, ct_averages[time == "3dpi", c("Cell_type", "Rel_freq")],
            by.x = "Type_t2", by.y = "Cell_type")
    colnames(alluvia_dt)[ncol(alluvia_dt)] <- "Rel_freq_t2"
    alluvia_dt <-
      merge(alluvia_dt, ct_averages[time == "Ctrl", c("Cell_type", "Rel_freq")],
            by.x = "Type_t1", by.y = "Cell_type")
    colnames(alluvia_dt)[ncol(alluvia_dt)] <- "Rel_freq_t1"
    
    alluvia_dt <- alluvia_dt[, {
      if(Rel_freq_t1 == 0){
        Rel_out_t1 = 0
      }else{
        Rel_out_t1 = t1t2_alluvium_size_t1/Rel_freq_t1
      }
      if(Rel_freq_t2 == 0){
        Rel_in_t2 = 0
        Rel_out_t2 = 0
      }else{
        Rel_in_t2 = t1t2_alluvium_size_t2/Rel_freq_t2
        Rel_out_t2 = t2t3_size_t2/Rel_freq_t2
      }
      if(Rel_freq_t3 == 0){
        Rel_in_t3 = 0
      }else{
        Rel_in_t3 = Transfer_t2t3/Rel_freq_t3
      }
      list(Rel_out_t1, Rel_in_t2, Rel_out_t2, Rel_in_t3)
    }, by = names(alluvia_dt)]
    # alluvia_dt <- alluvia_dt[, Alpha_t1t2 := max(Rel_out_t1, Rel_in_t2), by = names(alluvia_dt)]
    alluvia_dt <- alluvia_dt[, Alpha_t1t2 := ifelse(max(Rel_out_t1, Rel_in_t2) < 0.1, 0, 0.5), by = names(alluvia_dt)]
    alluvia_dt <- alluvia_dt[, Alpha_t2t3 := ifelse(max(Rel_out_t2, Rel_in_t3) < 0.1, 0, 0.5), by = names(alluvia_dt)]
  }else{
    alluvia_dt$Alpha_t1t2 <- 0.5
    alluvia_dt$Alpha_t2t3 <- 0.5
  }
  
  # Create alluvial datatable for plotting
  alluvia_dt_plot <- 
    rbind(alluvia_dt[, .(Cell_type = Type_t1,
                         Alluvium = Alluvium,
                         Fraction = t1t2t3_alluvium_size_t1,
                         Plot_alpha = Alpha_t1t2,
                         Time = 1)],
          alluvia_dt[, .(Cell_type = Type_t2,
                         Alluvium = Alluvium,
                         Fraction = t1t2t3_alluvium_size_t2,
                         Plot_alpha = Alpha_t2t3,
                         Time = 2)],
          alluvia_dt[, .(Cell_type = Type_t3,
                         Alluvium = Alluvium,
                         Fraction = t1t2t3_alluvium_size_t3,
                         Plot_alpha = 0.5,
                         Time = 3)])
  alluvia_dt_plot$Plot_alpha <- 0.5
  alluvia_dt_plot$Plot_alpha[alluvia_dt_plot$Fraction < fraction_threshold] <- 0
  
  return(alluvia_dt_plot)
}

ReadTransitions <- function(filename, edgelist_from, edgelist_to, sep = ","){
  transitions.wide <- fread(filename, sep = sep)
  colnames(transitions.wide)[1] <- "From"
  
  transitions.long <-
    melt(transitions.wide,
         id.vars = c("From"), variable.name = "To", 
         value.name = "Probability")
  transitions.long <- merge(transitions.long, edgelist_from[, c("Cell", "Node", "Cell.type")], 
                            by.x = "From", by.y = "Cell")
  colnames(transitions.long)[c(ncol(transitions.long) - 1, ncol(transitions.long))] <- c("Node_from", "Cell_type_from")
  transitions.long <- merge(transitions.long, edgelist_to[, c("Cell", "Node", "Cell.type")], 
                            by.x = "To", by.y = "Cell")
  colnames(transitions.long)[c(ncol(transitions.long) - 1, ncol(transitions.long))] <- c("Node_to", "Cell_type_to")

  return(transitions.long)
}

ParseRunName <- function(x){
  #x <- AUCs$Run_name[1]
  y <- unlist(strsplit(x, split = "_"))
  return(list(alpha = unlist(strsplit(y[3], split = "-"))[2], 
              epsilon = unlist(strsplit(y[4], split = "-"))[2], 
              beta = unlist(strsplit(y[5], split = "-"))[2],
              taua = unlist(strsplit(y[6], split = "-"))[2])) 
}

VaryingHyperparameters <- function(hp_list, baseline_hp, variable_hp){
  return(suppressWarnings(all(Map(`==`, hp_list[names(hp_list) != variable_hp], 
                                  baseline_hp[names(baseline_hp) != variable_hp]))))
}

ZoomAverages <- function(ct_and_transfer_averages, ct_of_interest, ratio_cutoff = 0.01){
  # The below compares anything to ct_of_interest and keeps it if it is above a threshold (0.01 relative)
  transfer_averages_zoom <- ct_and_transfer_averages$Transfer_ratios
  ct_averages <- ct_and_transfer_averages$Cell_type_frequencies
  transfer_averages_zoom <- 
    merge(transfer_averages_zoom, 
          ct_averages[, c("time", "Cell_type", "Rel_freq")], 
          by.x = c("t1_time", "Cell_type_from"),
          by.y = c("time", "Cell_type"))
  colnames(transfer_averages_zoom)[ncol(transfer_averages_zoom)] <- "Rel_freq_t1"
  transfer_averages_zoom <- 
    merge(transfer_averages_zoom, ct_averages, by.x = c("t2_time", "Cell_type_to"),
          by.y = c("time", "Cell_type"))
  colnames(transfer_averages_zoom)[ncol(transfer_averages_zoom)] <- "Rel_freq_t2"
  
  transfer_averages_zoom[, Keep_ct_t1 := Cell_type_from %in% ct_of_interest |
                           Cell_type_to %in% ct_of_interest & Transfer_ratio > ratio_cutoff * Rel_freq_t2,
                         by = names(transfer_averages_zoom)]
  transfer_averages_zoom <-
    transfer_averages_zoom[, Cell_type_from_zoom := ifelse(Keep_ct_t1, Cell_type_from, "Other"),
                           by = names(transfer_averages_zoom)]
  transfer_averages_zoom[, Keep_ct_t2 := Cell_type_to %in% ct_of_interest,
                         by = names(transfer_averages_zoom)]
  transfer_averages_zoom <-
    transfer_averages_zoom[, Cell_type_to_zoom := ifelse(Keep_ct_t2, Cell_type_to, "Other"),
                           by = names(transfer_averages_zoom)]
  # We can now aggregate the transfer averages but we have to update the cell type frequencies
  # as well. The ct_of_interest will keep their own frequencies, the other cell type frequencies
  # will be set to 0. 
  transfer_averages_zoom <-
    transfer_averages_zoom[, .(Transfer_ratio = sum(Transfer_ratio)), 
                           by = c("t1_time", "t2_time", "Cell_type_from_zoom", "Cell_type_to_zoom")]
  colnames(transfer_averages_zoom)[3:4] <- c("Cell_type_from", "Cell_type_to")
  ct_averages_zoom <- ct_averages
  ct_averages_zoom[, Cell_type_zoom := ifelse(Cell_type %in% ct_of_interest, Cell_type, "Other"),
                   by = names(ct_averages_zoom)]
  ct_averages_zoom <- ct_averages_zoom[, .(Rel_freq = sum(Rel_freq)), by = c("Cell_type_zoom", "time")]
  colnames(ct_averages_zoom)[1] <- "Cell_type"
  ct_add <- ct_averages[!(Cell_type %in% ct_of_interest), c("Cell_type", "time", "Rel_freq")]
  ct_add$Rel_freq <- 0
  ct_averages_zoom <- rbind(ct_averages_zoom, ct_add)
  
  return(list(Cell_type_frequencies = ct_averages_zoom, 
              Transfer_ratios = transfer_averages_zoom))
}

# Load and prep data ####
tree_list_in <- readRDS("~/Documents/Projects/heart_Bo/Data/Trees/Tree_list_oneEndo.rds")

annotations <- fread("~/Documents/Projects/heart_Bo/Data/final_metadata_Tmacromerged_2.csv")
timepoints <- unique(annotations[, c("orig.ident", "time")])
timepoints$Tree <-
  sapply(timepoints$orig.ident,
         function(x){unlist(strsplit(x, split = "[a-z,A-Z]$"))[1]})

celltype_colors_in <- read.csv("~/Documents/Projects/heart_Bo/Data/Cell_type_colors_2.csv",
                               stringsAsFactors = F)
celltype_colors <- setNames(celltype_colors_in$color, celltype_colors_in$Cell.type)
rm(celltype_colors_in)

rm(ct_freqs)
for(t in names(tree_list_in)){
  ct_freqs_tree <- setDT(tree_list_in[[t]]$Edge_list)[Cell.type != "NA", .(Freq = .N), by = "Cell.type"]
  ct_freqs_tree$Rel_freq <- ct_freqs_tree$Freq/sum(ct_freqs_tree$Freq)
  ct_freqs_tree$Tree <- t
  if(exists("ct_freqs")){
    ct_freqs <- rbind(ct_freqs, ct_freqs_tree)
  }else{
    ct_freqs <- ct_freqs_tree
  }
}
ct_freqs <-
  merge(ct_freqs, unique(timepoints[, c("time", "Tree")]),
        by.x = "Tree", by.y = "Tree")
ct_freqs <- ct_freqs[time %in% c("Ctrl", "3dpi", "7dpi")]
ct_freqs$time <- factor(ct_freqs$time, levels = c("Ctrl", "3dpi", "7dpi"))

moslin_data_path <- "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9/"
moslin_files <- data.table(Moslin_filename = list.files(path = moslin_data_path))
moslin_files <- moslin_files[, {
  filename_split <- unlist(strsplit(Moslin_filename, "_|-"))
  list(From = filename_split[2],
       To = filename_split[3],
       Alpha = as.numeric(filename_split[5]))},
  by = c("Moslin_filename")]
moslin_files <-
  moslin_files[, {
    if(grepl("H[0-9]", From)){
      from_time = "Ctrl"
      to_time = "3dpi"
    }else{
      from_time = "3dpi"
      to_time = "7dpi"
    }
    list(From_time = from_time, To_time = to_time)
  }, by = names(moslin_files)]

# Visualize couplings and frequencies ####
from_tree <- "H5"
to_tree <- "Hr27"

edgelist_from <- tree_list_in[[from_tree]]$Edge_list
edgelist_from <- edgelist_from[edgelist_from$Cell.type != "NA", ]
edgelist_from$Cell <- sapply(edgelist_from$to, function(x){substr(x, 3, nchar(x))})
colnames(edgelist_from)[1] <- "Node"

edgelist_to <- tree_list_in[[to_tree]]$Edge_list
edgelist_to <- edgelist_to[edgelist_to$Cell.type != "NA", ]
edgelist_to$Cell <- sapply(edgelist_to$to, function(x){substr(x, 3, nchar(x))})
colnames(edgelist_to)[1] <- "Node"

filename_pattern <- paste(from_tree, "_", to_tree, "_", sep = "")
moslin_filename <- moslin_files$Moslin_filename[grep(filename_pattern, moslin_files$Moslin_filename)]
moslin_filepath <- paste(moslin_data_path, moslin_filename, sep = "")

transitions.long <- 
  ReadTransitions(filename = moslin_filepath,
                  edgelist_from = edgelist_from, edgelist_to = edgelist_to, sep = ",")
transitions.wide <- fread(moslin_filepath, sep = ",")
min_value <- min(transitions.long$Probability[transitions.long$Probability > 0])
transitions_test <- data.frame(transitions.wide)[, -1]#[1:200, 2:201])
rownames(transitions_test) <- transitions.wide$V1 #[1:200]
transitions_test[transitions_test == 0] <- min_value
edgelist_from <- edgelist_from[order(edgelist_from$Cell.type), ]
edgelist_from <- edgelist_from[edgelist_from$Cell %in% rownames(transitions_test), ]
edgelist_to <- edgelist_to[order(edgelist_to$Cell.type), ] #[edgelist_to$Cell %in% colnames(transitions_test)]
edgelist_to <- edgelist_to[edgelist_to$Cell %in% colnames(transitions_test), ]
transitions_test <- transitions_test[edgelist_from$Cell, edgelist_to$Cell]

pheatmap_anno <- data.frame(Cell_type = annotations$Cell_type)
rownames(pheatmap_anno) <- annotations$Cell
row_anno <- pheatmap_anno[rownames(transitions_test), , drop=F]
col_anno <- pheatmap_anno[colnames(transitions_test), , drop=F]
ann_colors = list(Cell_type = celltype_colors[unique(c(col_anno$Cell_type, row_anno$Cell_type))])

# Fig 4b and part of Figure 5.2
pheatmap_filename <- paste("./Images/Couplings_pheatmap_", moslin_filename, ".png", sep = "")
# png(pheatmap_filename, width = 450, height = 350)
pheatmap::pheatmap(log10(transitions_test), fontsize = 24, legend = T, #clustering_method = "ward.D2",
                   cluster_rows = F, cluster_cols = F,
                   show_rownames = F, show_colnames = F,
                   annotation_row = row_anno,
                   annotation_col = col_anno,
                   annotation_names_col = F, annotation_names_row = F,
                   annotation_colors = ann_colors, annotation_legend = F)
# dev.off()

mean_transfers <- transitions.long[, .(Transfer_ratio = sum(Probability)), by = c("Cell_type_from", "Cell_type_to")]
mean_transfers_wide <- dcast(mean_transfers, Cell_type_from ~ Cell_type_to, value.var = "Transfer_ratio")
mean_transfers_mat <- as.matrix(mean_transfers_wide[, -1])
rownames(mean_transfers_mat) <- mean_transfers_wide$Cell_type_from

pheatmap_anno <- data.frame(Cell_type = union(mean_transfers$Cell_type_from, mean_transfers$Cell_type_to))
rownames(pheatmap_anno) <- pheatmap_anno$Cell_type
row_anno <- pheatmap_anno[rownames(mean_transfers_mat), , drop=F]
col_anno <- pheatmap_anno[colnames(mean_transfers_mat), , drop=F]
ann_colors = list(Cell_type = celltype_colors[unique(c(col_anno$Cell_type, row_anno$Cell_type))])

# Part of Figure S5.2
# png(paste("../../../Images/Mean_couplings_pheatmap_", moslin_filename, ".png", sep = ""),
#     width = 450, height = 350)
pheatmap::pheatmap(log10(mean_transfers_mat), fontsize = 24, legend = T, 
                   cluster_rows = F, cluster_cols = F,
                   show_rownames = F, show_colnames = F,
                   annotation_row = row_anno,
                   annotation_col = col_anno,
                   annotation_names_col = F, annotation_names_row = F,
                   annotation_colors = ann_colors, annotation_legend = F)
# dev.off()

# Part of Figure S5.2
# png("./Images/Ct_freqs_H5_Hr27.png",
#     width = 350, height = 350)
ggplot(ct_freqs[Tree %in% c("H5", "Hr27")]) +
  geom_bar(aes(x = Tree, y = Rel_freq, fill = Cell.type), stat = "identity", width = 0.6) +
  scale_x_discrete(expand = c(0, 0), breaks = c("H5", "Hr27"), labels = c("Ctrl", "3dpi")) + 
  scale_y_continuous(expand = expansion(mult = c(0, 0), add = c(0, 0.1))) +
  labs(x = "", y = "Cell type fraction") + 
  scale_fill_manual(values = celltype_colors) +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 32),
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 18),
        panel.background = element_blank())
# dev.off()

transfer_ratios_trees <- 
  fread("../../../Data/All_celltype_transfer_ratios_with_background_tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9")
  
ct_and_transfer_averages <- AverageFrequenciesAndTransferRatios(transfer_ratios = transfer_ratios_trees,
                                                                tree_times = unique(ct_freqs[, c("Tree", "time")]))

transfer_av_ctrl_3dpi <- ct_and_transfer_averages$Transfer_ratios[t1_time == "Ctrl"]
transfer_av_ctrl_3dpi_wide <- 
  dcast(transfer_av_ctrl_3dpi, Cell_type_from ~ Cell_type_to, value.var = "Transfer_ratio")
transfer_av_ctrl_3dpi_mat <- as.matrix(transfer_av_ctrl_3dpi_wide[, -1])
rownames(transfer_av_ctrl_3dpi_mat) <- transfer_av_ctrl_3dpi_wide$Cell_type_from
transfer_av_ctrl_3dpi_mat <- transfer_av_ctrl_3dpi_mat[rowSums(transfer_av_ctrl_3dpi_mat) > 1e-10, ]
transfer_av_ctrl_3dpi_mat <- transfer_av_ctrl_3dpi_mat[, colSums(transfer_av_ctrl_3dpi_mat) > 1e-10]
transfer_av_ctrl_3dpi_mat[transfer_av_ctrl_3dpi_mat < 1e-10] <- 1e-10

pheatmap_anno <- data.frame(Cell_type = union(transfer_av_ctrl_3dpi$Cell_type_from, transfer_av_ctrl_3dpi$Cell_type_to))
rownames(pheatmap_anno) <- pheatmap_anno$Cell_type
row_anno <- pheatmap_anno[rownames(transfer_av_ctrl_3dpi_mat), , drop=F]
col_anno <- pheatmap_anno[colnames(transfer_av_ctrl_3dpi_mat), , drop=F]
ann_colors = list(Cell_type = celltype_colors[unique(c(col_anno$Cell_type, row_anno$Cell_type))])

# Part of Figure S5.2
# png("../../../Images/Moscot_mean_couplings_tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9.png",
#     width = 450, height = 350)
pheatmap::pheatmap(log10(transfer_av_ctrl_3dpi_mat), fontsize = 24, legend = T, 
                   cluster_rows = F, cluster_cols = F,
                   show_rownames = F, show_colnames = F,
                   annotation_row = row_anno,
                   annotation_col = col_anno,
                   annotation_names_col = F, annotation_names_row = F,
                   annotation_colors = ann_colors, annotation_legend = F)
# dev.off()

ct_freqs_ctrl_3dpi <- ct_and_transfer_averages$Cell_type_frequencies[time %in% c("Ctrl", "3dpi")]

# Part of Figure S5.2
# png("./Images/Ct_freqs_ctrl_3dpi.png",
#     width = 350, height = 350)
ggplot(ct_freqs_ctrl_3dpi) +
  geom_bar(aes(x = time, y = Rel_freq, fill = Cell_type), stat = "identity", width = 0.6) +
  scale_x_discrete(expand = c(0, 0)) +#, breaks = c("H5", "Hr27"), labels = c("Ctrl", "3dpi")) + 
  scale_y_continuous(expand = expansion(mult = c(0, 0), add = c(0, 0.1))) +
  labs(x = "", y = "Cell type fraction") + 
  scale_fill_manual(values = celltype_colors) +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 32),
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 18),
        panel.background = element_blank())
# dev.off()

# Cell type-cell type persistency test ####
data_paths <- c("/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.6_epsilon-0.01_beta-0.2_taua-0.9/",
                "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.5_epsilon-0.005_beta-0.2_taua-0.9/",
                "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9/",
                "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.4_epsilon-0.01_beta-0.2_taua-0.9/",
                "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.1_taua-0.9/",
                "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.3_taua-0.9/",
                "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.95/",
                "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.85/",
                "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.5_epsilon-0.1_beta-0.2_taua-0.9/"
                )
rm(files_dt)
for(i in 1:length(data_paths)){
  if(exists("files_dt")){
    files_dt <- rbind(files_dt,
                      data.table(Path = data_paths[i],
                                 Filename = list.files(path = data_paths[i])))
  }else{
    files_dt <- data.table(Path = data_paths[i],
                           Filename = list.files(path = data_paths[i]))
  }
}
files_dt$Run_name <-
  sapply(files_dt$Path,
         function(x){
           y <- unlist(strsplit(x = x, split = "/"))
           return(y[length(y)])
         }
  )
files_dt$F1_score <- NA
files_dt$F1_support <- NA

ct_persistency_results <- list()
ct_persistency_results_ovr <- list()

for(i in 1:nrow(files_dt)){
  this_run <- files_dt$Run_name[i]
  filename <- files_dt$Filename[i]
  file_name_split <- unlist(strsplit(filename, "-|_"))
  dataset_t1 <- file_name_split[2]
  dataset_t2 <- file_name_split[3]
  t1 <- tree_time$time[tree_time$Tree == dataset_t1]
  print(paste("t1: ", dataset_t1, ", t2: ", dataset_t2, ", run: ", this_run))
  
  # Create edgelists and load dataset
  # print("Edgelists")
  edgelist_from <- tree_list_in[[dataset_t1]]$Edge_list
  edgelist_from <- edgelist_from[edgelist_from$Cell.type != "NA", ]
  edgelist_from$Cell <- sapply(edgelist_from$to, function(x){substr(x, 3, nchar(x))})
  colnames(edgelist_from)[1] <- "Node"
  edgelist_to <- tree_list_in[[dataset_t2]]$Edge_list
  edgelist_to <- edgelist_to[edgelist_to$Cell.type != "NA", ]
  edgelist_to$Cell <- sapply(edgelist_to$to, function(x){substr(x, 3, nchar(x))})
  colnames(edgelist_to)[1] <- "Node"
  print("Loading")
  transitions.long <- ReadTransitions(filename = paste(files_dt$Path[i], files_dt$Filename[i], sep = ""),
                                       edgelist_from = edgelist_from,
                                      edgelist_to = edgelist_to, sep = ",")

  print("Cell type selection")
  cell_types_t1 <- transitions.long[, .(Freq = uniqueN(From)), by = c("Cell_type_from")]
  cell_types_t1 <- cell_types_t1[Freq > 10]
  cell_types_t2 <- transitions.long[, .(Freq = uniqueN(To)), by = c("Cell_type_to")]
  cell_types_t2 <- cell_types_t2[Freq > 10]
  transitions_suff_freq <- transitions.long[Cell_type_from %in% cell_types_t1$Cell_type_from &
                                              Cell_type_to %in% cell_types_t2$Cell_type_to]

  # Coupling visualization - only for manual use
  if(F){
    coupling_mat <- dcast(transitions_suff_freq, 
                          From ~ To, value.var = "Probability")
    cells_from <- coupling_mat$From
    coupling_mat <- as.matrix(coupling_mat[, -1])
    rownames(coupling_mat) <- cells_from
    
    edgelist_from <- edgelist_from[order(edgelist_from$Cell.type), ]
    edgelist_from <- edgelist_from[edgelist_from$Cell %in% rownames(coupling_mat), ]
    edgelist_to <- edgelist_to[order(edgelist_to$Cell.type), ]
    edgelist_to <- edgelist_to[edgelist_to$Cell %in% colnames(coupling_mat), ]
    coupling_mat <- coupling_mat[edgelist_from$Cell, edgelist_to$Cell]
    
    pheatmap_anno <- data.frame(Cell_type = annotations$Cell_type)
    rownames(pheatmap_anno) <- annotations$Cell
    row_anno <- pheatmap_anno[rownames(coupling_mat), , drop=F]
    col_anno <- pheatmap_anno[colnames(coupling_mat), , drop=F]
    ann_colors = list(Cell_type = celltype_colors[unique(c(col_anno$Cell_type, row_anno$Cell_type))])
    # png(paste("./Images/Moscot_couplings_", dataset_t1, "_", dataset_t2, "_", this_run, ".png", sep = ""),
    #      width = 1.6, height = 1.2, res = 900, units = "in", type = "cairo")
    pheatmap::pheatmap(log10(coupling_mat), fontsize = 8, legend = T, #clustering_method = "ward.D2",
                       cluster_rows = F, cluster_cols = F,
                       show_rownames = F, show_colnames = F,
                       annotation_row = row_anno,
                       annotation_col = col_anno,
                       annotation_names_col = F, annotation_names_row = F,
                       annotation_colors = ann_colors, annotation_legend = F)
    # dev.off()
    
    focus_type_to <- "Endocardium (Atrium)"
    focus_type_from <- "Macrophages"
    
    # Distribution plots in 4b
    ggplot() +
      geom_histogram(data = transitions_suff_freq[transitions_suff_freq$Cell_type_to == focus_type_to &
                                               transitions_suff_freq$Cell_type_from != focus_type_from,],
                     aes(x = log10(Probability)), fill = "grey") +
      geom_histogram(data = transitions_suff_freq[transitions_suff_freq$Cell_type_to == focus_type_to &
                                                    transitions_suff_freq$Cell_type_from == focus_type_from,],
                     aes(x = log10(Probability)), fill = ann_colors$Cell_type[focus_type_from], alpha = 0.75) +
      scale_y_continuous(breaks = c(0, 50000, 100000), #c(0, 20000, 40000, 60000),
                         labels = c(0, expression("50 x 10"^3), 
                                    expression("100 x 10"^3))) +
      theme(plot.title = element_blank(), #text(hjust = 0.5),
            panel.grid = element_blank(),
            text = element_text(size = 7, family = "Helvetica Neue"),
            panel.background = element_rect(fill = NA, colour = ann_colors$Cell_type[focus_type_to]),
            axis.text = element_text(size = 6),
            axis.text.x = element_text(vjust = 0),
            axis.title.x = element_text(margin = margin(5, 0, 0, 0)),
            strip.background = element_rect(fill = NA),
            strip.text = element_text(size = 7),
            legend.key=element_blank(),
            panel.spacing = unit(1, "lines"),
            plot.margin = unit(c(0.2, 0.5, 0, 0.5), "lines"))
    # ggsave(paste("./Images/Moscot_noiter_EndoA_Macroph_distribution_H5_Hr27_", this_run, ".png", sep = ""),
    #         width = 1.4, height = 0.8, units = "in", device = png, dpi = 900, type = "cairo")
  }
  
  print("Testing")
  # All vs. all
  ct_test_grid <- data.table(expand.grid(Cell_type_from = cell_types_t1$Cell_type_from,
                         Cell_type_to = cell_types_t2$Cell_type_to))
  ct_test_grid$Data_t1 <- dataset_t1
  ct_test_grid$Data_t2 <- dataset_t2
  ct_test_grid$t1 <- t1
  ct_test_grid$Welch.p <- NA
  for(type_t2 in cell_types_t2$Cell_type_to){
    transitions_type_t2 <- transitions_suff_freq[Cell_type_to == type_t2]
    for(type_t1 in cell_types_t1$Cell_type_from){
      j <- which(ct_test_grid$Cell_type_from == type_t1 & ct_test_grid$Cell_type_to == type_t2)
      if(length(j) > 1){
        print("Multiple indices found.")
        next
      }
      x <- transitions_type_t2$Probability[transitions_type_t2$Cell_type_from == type_t1]
      y <- transitions_type_t2$Probability[transitions_type_t2$Cell_type_from != type_t1]
      if(sd(x) == 0 && sd(y) == 0){
        next
      }else{
        ct_test_grid$Welch.p[j] <-
          t.test(x, y, alternative = "greater")$p.value
      }
    }
  }
  
  # One vs. rest - how good are we at predicting a certain t2 type? Positive outcome is
  # t1 type == t2 type.
  ct_test_grid_same_ovr <-
    data.table(Cell_type_to = intersect(cell_types_t1$Cell_type_from, cell_types_t2$Cell_type_to),
               Cell_type_from = "Same",
               Data_t1 = dataset_t1,
               Data_t2 = dataset_t2,
               t1 = t1,
               Welch.p = NA)
  ct_test_grid_rest_ovr <-
    data.table(Cell_type_to = intersect(cell_types_t1$Cell_type_from, cell_types_t2$Cell_type_to),
               Cell_type_from = "Rest",
               Data_t1 = dataset_t1,
               Data_t2 = dataset_t2,
               t1 = t1,
               Welch.p = NA)
  
  for(k in 1:nrow(ct_test_grid_same_ovr)){
    type_t2 <- ct_test_grid_same_ovr$Cell_type_to[k]
    transitions_type_t2 <- transitions_suff_freq[Cell_type_to == type_t2]
    x <- transitions_type_t2$Probability[transitions_type_t2$Cell_type_from == type_t2]
    y <- transitions_type_t2$Probability[transitions_type_t2$Cell_type_from != type_t2]
    if(sd(x) == 0 && sd(y) == 0){
      next
    }else{
      ct_test_grid_same_ovr$Welch.p[k] <-
        t.test(x, y, alternative = "greater")$p.value
      ct_test_grid_rest_ovr$Welch.p[k] <-
        t.test(x, y, alternative = "less")$p.value
    }
  }
  ct_test_grid_ovr <- rbind(ct_test_grid_same_ovr, ct_test_grid_rest_ovr)
    
  # F1 testing. First compute for the marginal coupling of t1 cells to t2 cell types by summing
  # over t2 cells. For each t1 cell, select the argmax t2 type. For a fixed A, true positives 
  # are t1 type A cells whose argmax type is A, false positives are t1 cells that are not type A
  # but whose argmax type is A, false negatives are t1 cells that are type A but whose argmax is
  # not, and support is the number of t1 type A cells.
  # Then calculate for type precision, recall, and F1:
  # Precision is TP/(TP + FP), recall is TP/(TP + FN), F1 is 2 * Precision * Recall/(Precision + Recall).
  # Finally, calculate the support-weighted averages of the F1 scores.
  # For the AUROCs, we constrain ourselves to the same set of cell types at t1 and t2 relatively
  # late, after running the Welch tests and outside the loop. For F1 scores, we can do this earlier, 
  # namely here.
  if(F){
    shared_types <- intersect(cell_types_t1$Cell_type_from, cell_types_t2$Cell_type_to)
    transitions_filter <-
      transitions_suff_freq[Cell_type_from %in% shared_types & Cell_type_to %in% shared_types]
    transition_sums <-
      transitions_filter[, .(Sum_prob = sum(Probability)), by = list(From, Cell_type_from, Cell_type_to)]
    persistence_predictions <-
      transition_sums[, .(Predicted_to = Cell_type_to[Sum_prob == max(Sum_prob)]), by = list(From, Cell_type_from)]
    F1_scores <-
      merge(persistence_predictions[, .(Support = .N,
                                        TP = sum(Cell_type_from == Predicted_to),
                                        FN = sum(Cell_type_from != Predicted_to)), 
                                    by = list(Cell_type = Cell_type_from)],
            persistence_predictions[, .(FP = sum(Cell_type_from != Predicted_to)), 
                                    by = list(Cell_type = Predicted_to)])
    F1_scores[, c("Precision", "Recall") := 
                list(TP/(TP + FP), 
                     TP/(TP + FN)), by = Cell_type]
    F1_scores$F1 <- 2 * F1_scores$Precision * F1_scores$Recall/(F1_scores$Precision + F1_scores$Recall)
    files_dt$F1_score[i] <- weighted.mean(x = F1_scores$F1, w = F1_scores$Support)
    files_dt$F1_support[i] <- sum(F1_scores$Support)
  }
  
  print("Integration")
  # Output: list of test grids; every list item is a set of hyperparameters and has the path name as key. 
  # The test grids have information on to and from trees and their t1 timepoints, and then of course the cell types and the
  # Welch test probabilities.
  # For one-vs-one
  if(this_run %in% names(ct_persistency_results)){
    # Add test grid to existing test grid
    ct_persistency_results[[this_run]] <- rbind(ct_persistency_results[[this_run]], ct_test_grid)
  }else{
    # Instantiate list element with test grid
    ct_persistency_results[[this_run]] <- ct_test_grid
  }
  # For one-vs-rest
  if(this_run %in% names(ct_persistency_results_ovr)){
    # Add test grid to existing test grid
    ct_persistency_results_ovr[[this_run]] <- rbind(ct_persistency_results_ovr[[this_run]], ct_test_grid_ovr)
  }else{
    # Instantiate list element with test grid
    ct_persistency_results_ovr[[this_run]] <- ct_test_grid_ovr
  }
}

# Save results
# for(run_name in names(ct_persistency_results)){
#   write_file_path <- paste("../../../Data/CT_welch_for_ROC", run_name, sep = "_")
#   fwrite(ct_persistency_results[[run_name]], file = write_file_path)
# }
# for(run_name in names(ct_persistency_results_ovr)){
#   write_file_path <- paste("../../../Data/CT_welch_for_ROC_OvR", run_name, sep = "_")
#   fwrite(ct_persistency_results_ovr[[run_name]], file = write_file_path)
# }

# Evaluation of F1 testing results. Set flag to True if wanted.
if(F){
  files_dt <- files_dt[complete.cases(files_dt), ]
  weighted.mean(x = files_dt$F1_score, w = files_dt$F1_support)
  # 0.67.
}

# AUROCs for OvR analysis
moslin_path_name <- "tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9"
read_file_path <- paste("../../../Data/CT_welch_for_ROC_OvR", moslin_path_name, sep = "_")
ROC_OvR_df <- fread(read_file_path)
ROC_OvR_df$Expected_out <- c(0,1)[as.integer(ROC_OvR_df$Cell_type_from != "Same") + 1]
ROC_OvR_df$time <-
  with(ROC_OvR_df, factor(t1, 
                      levels = c('Ctrl', '3dpi'), labels = c('3dpi', '7dpi')))

Welch_ROC_OvR_plot <- 
  ggplot(ROC_OvR_df, 
         aes(d = Expected_out, m = Welch.p)) + 
  facet_wrap(~time) +
  geom_roc(n.cuts = 0, size = 0.5) 
AUC_OvR_values <- data.frame(calc_auc(Welch_ROC_OvR_plot))
colnames(AUC_OvR_values)[1] <- "time"
AUC_OvR_values$time <- c("3dpi", "7dpi")
AUC_OvR_values$y <- rep(0.85, 2)
Welch_ROC_OvR_plot +
  labs(x = "FPR", y = "TPR") + #, color = "") + 
  geom_text(data = AUC_OvR_values, inherit.aes= F,show.legend = FALSE, x = 0.5, 
            aes(label = paste("AUC = ", round(AUC, 3), sep = ""), y = y),
            size = 2.5) +
  scale_x_continuous(breaks = c(0, 0.5, 1)) + #, expand = c(0, 0)) +
  scale_y_continuous() + #expand = c(0, 0)) +
  theme(plot.title = element_blank(), #text(hjust = 0.5),
        panel.grid = element_blank(),
        text = element_text(size = 7, family = "Helvetica Neue"),
        panel.background = element_rect(fill = NA, colour = "black"),
        axis.text = element_text(size = 6),
        axis.text.x = element_text(vjust = 0),
        axis.title.x = element_text(margin = margin(5, 0, 0, 0)),
        strip.background = element_rect(fill = NA),
        strip.text = element_text(size = 7),
        legend.key=element_blank(),
        panel.spacing = unit(1, "lines"),
        plot.margin = unit(c(0, 0.5, 0, 0.5), "lines"))
ggsave(paste("../../../Images/Welch_ROC_OvR_", moslin_path_name, ".png", sep = ""),
       width = 3, height = 1.4, units = "in", device = png, dpi = 900, type = "cairo")

# AUROCs for OvO analysis
moslin_path_name <- "tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9"
read_file_path <- paste("../../../Data/CT_welch_for_ROC", moslin_path_name, sep = "_")
top_freq_types_full <- fread(read_file_path)
ROC_df <- 
  top_freq_types_full[top_freq_types_full[, 
                                          .I[which(Cell_type_to %in% intersect(Cell_type_from, Cell_type_to) &
                                                     Cell_type_from %in% intersect(Cell_type_from, Cell_type_to))], by = c("t1")]$V1]
ROC_df$Cell_type_from <- factor(x = ROC_df$Cell_type_from, levels = intersect(top_freq_types_full$Cell_type_from, top_freq_types_full$Cell_type_to))
ROC_df$Cell_type_to <- factor(x = ROC_df$Cell_type_to, levels = intersect(top_freq_types_full$Cell_type_from, top_freq_types_full$Cell_type_to))
ROC_df$Expected_out <- c(0, 1)[as.integer(ROC_df$Cell_type_to != ROC_df$Cell_type_from) + 1]
ROC_df$time <-
  with(ROC_df, factor(t1, 
                      levels = c('Ctrl', '3dpi'), labels = c('3dpi', '7dpi')))

# Figure 4c
Welch_ROC_plot <- 
  ggplot(ROC_df, 
         aes(d = Expected_out, m = Welch.p)) + 
  facet_wrap(~time) +
  geom_roc(n.cuts = 0, size = 0.5) 
AUC_values <- data.frame(calc_auc(Welch_ROC_plot))
colnames(AUC_values)[1] <- "time"
AUC_values$time <- c("3dpi", "7dpi")
AUC_values$y <- rep(0.85, 2)
Welch_ROC_plot +
  labs(x = "FPR", y = "TPR") + #, color = "") + 
  geom_text(data = AUC_values, inherit.aes= F,show.legend = FALSE, x = 0.5, 
            aes(label = paste("AUC = ", round(AUC, 3), sep = ""), y = y),
            size = 2.5) +
  scale_x_continuous(breaks = c(0, 0.5, 1)) + #, expand = c(0, 0)) +
  scale_y_continuous() + #expand = c(0, 0)) +
  theme(plot.title = element_blank(), #text(hjust = 0.5),
        panel.grid = element_blank(),
        text = element_text(size = 7, family = "Helvetica Neue"),
        panel.background = element_rect(fill = NA, colour = "black"),
        axis.text = element_text(size = 6),
        axis.text.x = element_text(vjust = 0),
        axis.title.x = element_text(margin = margin(5, 0, 0, 0)),
        strip.background = element_rect(fill = NA),
        strip.text = element_text(size = 7),
        legend.key=element_blank(),
        panel.spacing = unit(1, "lines"),
        plot.margin = unit(c(0, 0.5, 0, 0.5), "lines"))
ggsave(paste("./Images/Welch_ROC_", moslin_path_name, ".png", sep = ""),
       width = 3, height = 1.4, units = "in", device = png, dpi = 900, type = "cairo")

# Different hyperparameter values
# Figure S5.1
for(run_name in unique(files_dt$Run_name)){
  read_file_path <- paste("../../../Data/CT_welch_for_ROC", run_name, sep = "_")
  ct_persistency_results[[run_name]] <- fread(read_file_path)
}
run_names <- names(ct_persistency_results)
AUCs <- data.table(Run_name = rep(run_names, each = 2),
                   AUC = NA,
                   Time = rep(c("3dpi", "7dpi"), times = length(run_names))
)
for(i in 1:nrow(AUCs)){
  run_name <- AUCs$Run_name[i]
  run_time <- AUCs$Time[i]

  ROC_df <- ct_persistency_results[[run_name]]
  ROC_df <- ROC_df[ROC_df[, .I[which(Cell_type_to %in% intersect(Cell_type_from, Cell_type_to) &
                                Cell_type_from %in% intersect(Cell_type_from, Cell_type_to))], by = c("t1")]$V1]
  ROC_df$Expected_out <- c(0, 1)[as.integer(ROC_df$Cell_type_to != ROC_df$Cell_type_from) + 1]
  ROC_df$time <-
    with(ROC_df, factor(t1,
                        levels = c('Ctrl', '3dpi'), labels = c('3dpi', '7dpi')))
  ROC_df <- ROC_df[ROC_df$time == run_time, ]
  Welch_ROC_plot <- 
    ggplot(ROC_df,
           aes(d = Expected_out, m = Welch.p)) +
    geom_roc(n.cuts = 0, size = 3)
  AUCs$AUC[i] <- calc_auc(Welch_ROC_plot)$AUC
}

AUCs[, c("alpha", "epsilon", "beta", "taua") := ParseRunName(Run_name), by = Run_name]
baseline_hyperparameters = list(alpha = 0.5, epsilon = 0.01, beta = 0.2, taua = 0.9)

alpha_AUCs <-
  AUCs[apply(AUCs, 1, function(x){
    hp_list = list(alpha = x[4], epsilon = x[5], beta = x[6], taua = x[7])
    VaryingHyperparameters(hp_list, baseline_hyperparameters, "alpha")
  }), ,]
g <- ggplot(alpha_AUCs) +
  geom_bar(aes(x = alpha, y = AUC, alpha = Time, fill = as.numeric(alpha)), 
           stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.5, 1))+
  labs(x = "\u03B1") +
  scale_alpha_manual(values = c(0.4, 0.7)) + 
  scale_fill_gradient2(midpoint = 0.45, guide = "none") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 12),
        panel.background = element_rect(fill = NA),
        panel.border = element_rect(colour = "black", fill = NA, size = 2),
        strip.background = element_rect(fill = NA),
        legend.key=element_blank())
g
ggsave("../../../Images/Welch_alpha_varied_AUCs.png",
       width = 3, height = 2, units = "in", device = png, dpi = 900, type = "cairo")

# Extract colors
b <- ggplot_build(g)
hyperpar_colours <- b$data[[1]][, c("x", "fill")]
hyperpar_colours <- hyperpar_colours[order(hyperpar_colours$x), ]

epsilon_AUCs <-
  AUCs[apply(AUCs, 1, function(x){
    hp_list = list(alpha = x[4], epsilon = x[5], beta = x[6], taua = x[7])
    VaryingHyperparameters(hp_list, baseline_hyperparameters, "epsilon")
  }), ,]
ggplot(epsilon_AUCs) +
  geom_bar(aes(x = epsilon, y = AUC, alpha = Time, fill = epsilon), 
           stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.5, 1))+
  labs(x = "\u03B5") +
  scale_alpha_manual(values = c(0.4, 0.7)) + 
  scale_fill_manual(values = unique(hyperpar_colours$fill), guide = "none") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 12),
        panel.background = element_rect(fill = NA),
        panel.border = element_rect(colour = "black", fill = NA, size = 2),
        strip.background = element_rect(fill = NA),
        legend.key=element_blank())
ggsave("../../../Images/Welch_epsilon_varied_AUCs.png",
       width = 3, height = 2, units = "in", device = png, dpi = 900, type = "cairo")

beta_AUCs <-
  AUCs[apply(AUCs, 1, function(x){
    hp_list = list(alpha = x[4], epsilon = x[5], beta = x[6], taua = x[7])
    VaryingHyperparameters(hp_list, baseline_hyperparameters, "beta")
  }), ,]
ggplot(beta_AUCs) +
  geom_bar(aes(x = beta, y = AUC, fill = beta, alpha = Time), 
           stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.5, 1))+
  labs(x = "\u03B2") +
  scale_alpha_manual(values = c(0.4, 0.7)) + 
  scale_fill_manual(values = unique(hyperpar_colours$fill), guide = "none") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 12),
        panel.background = element_rect(fill = NA),
        panel.border = element_rect(colour = "black", fill = NA, size = 2),
        strip.background = element_rect(fill = NA),
        legend.key=element_blank())
ggsave("../../../Images/Welch_beta_varied_AUCs.png",
       width = 3, height = 2, units = "in", device = png, dpi = 900, type = "cairo")

taua_AUCs <-
  AUCs[apply(AUCs, 1, function(x){
    hp_list = list(alpha = x[4], epsilon = x[5], beta = x[6], taua = x[7])
    VaryingHyperparameters(hp_list, baseline_hyperparameters, "taua")
  }), ,]
ggplot(taua_AUCs) +
  geom_bar(aes(x = taua, y = AUC, fill = taua, alpha = Time), 
           stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.5, 1))+
  labs(x = "\u03C4") +
  scale_alpha_manual(values = c(0.4, 0.7)) + 
  scale_fill_manual(values = unique(hyperpar_colours$fill), guide = "none") +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 12),
        panel.background = element_rect(fill = NA),
        panel.border = element_rect(colour = "black", fill = NA, size = 2),
        strip.background = element_rect(fill = NA),
        legend.key=element_blank())
ggsave("../../../Images/Welch_taua_varied_AUCs.png",
       width = 3, height = 2, units = "in", device = png, dpi = 900, type = "cairo")
# End of plotting S5.1b

# Subsample the ROC curves and AUCs:
# Inset figure 4c
t1 <- "Ctrl"
t2 <- "3dpi"

# Loop over m1 and m2
tot_samples <- 0
AUC_sample <- 
  data.table(Dataset_combinations = integer(),
            t1_sets = character(),
            t2_sets = character(),
            AUC = numeric())
for(m1 in 1:length(tree_time$Tree[tree_time$time == t1])){
  t1_trees_combinations <- combn(tree_time$Tree[tree_time$time == t1], m1)
  print(paste("T1 dataset count: ", m1, ", ", ncol(t1_trees_combinations), " combination(s).", sep = ""))
  for(m2 in 1:length(tree_time$Tree[tree_time$time  == t2])){
    t2_trees_combinations <- combn(tree_time$Tree[tree_time$time == t2], m2)
    print(paste("T2 dataset count: ", m2, ", ", ncol(t2_trees_combinations), " combination(s).", sep = ""))
    for(t1_c in 1:ncol(t1_trees_combinations)){
      t1_tree_string <- paste(t1_trees_combinations[, t1_c], collapse = "_")
      for(t2_c in 1:ncol(t2_trees_combinations)){
        t2_tree_string <- paste(t2_trees_combinations[, t2_c], collapse = "_")
        this_AUC <- calc_auc(ggplot(ROC_df[ROC_df$Data_t1 %in% t1_trees_combinations[, t1_c] &
                                             ROC_df$Data_t2 %in% t2_trees_combinations[, t2_c]],
                                    aes(d = Expected_out, m = Welch.p)) +
                               geom_roc())$AUC

        AUC_sample_add <-
          data.table(Dataset_combinations = m1 * m2,
                    t1_sets = t1_tree_string,
                    t2_sets = t2_tree_string,
                    AUC = this_AUC)
        AUC_sample <- rbind(AUC_sample, AUC_sample_add)
        }
    }
    tot_samples <- tot_samples + ncol(t1_trees_combinations) * ncol(t2_trees_combinations)
  }
}
# fwrite(AUC_sample, paste("./Data/Same_cell_subsampled_ctrl_3dpi_", moslin_path_name, sep = ""))
# fwrite(AUC_sample, paste("./Data/Same_cell_subsampled_3dpi_7dpi_", moslin_path_name, sep = ""))

ggplot(AUC_sample) + 
  geom_histogram(aes(x = AUC), bins = 60) +
  theme(text = element_text(family = "Helvetica Neue", size = 6),
        panel.grid = element_blank(),
        axis.ticks = element_line(size = 0.25),
        axis.title.x = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)),
        # axis.title.y = element_blank(),
        panel.background = element_rect(fill = NA, color = "black"),
        plot.background = element_blank(),
        panel.border = element_rect(color = "black", fill = NA),
        plot.margin = unit(c(0.25, 0.25, 0, 0), units = "line")) +
  scale_x_continuous(expand = c(0, 0), breaks = c(0.97, 0.99), labels = c(0.97, 0.99), limits = c(0.96, 1)) +
  scale_y_continuous(expand = c(0, 0)) #, limits = c(-10, 8500), breaks = c(0, 8000)) +

# ggsave(paste("./Images/Subsampled_", t2, "_Welch_AUC_distribution_", moslin_path_name, ".png", sep = ""),
# End inset figure 4c

# Path towards transient fibroblasts ####
freq_cutoff <- 10
moslin_data_path <- 
  "/Volumes/Data 1/Current Project Data/Moscot/tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9/"
moslin_files <- data.table(Moslin_filename = list.files(path = moslin_data_path))
moslin_files <- moslin_files[, {
  filename_split <- unlist(strsplit(Moslin_filename, "_|-"))
  list(From = filename_split[2],
       To = filename_split[3],
       Alpha = as.numeric(filename_split[5]))},
  by = c("Moslin_filename")]
moslin_files <-
  moslin_files[, {
    if(grepl("H[0-9]", From)){
      from_time = "Ctrl"
      to_time = "3dpi"
    }else{
      from_time = "3dpi"
      to_time = "7dpi"
    }
    list(From_time = from_time, To_time = to_time)
  }, by = names(moslin_files)]

rm(transitions_top)
rm(inout_ratios)
rm(transfer_ratios)
for(i in 1:nrow(moslin_files)){
  from_tree <- moslin_files$From[i]
  to_tree <- moslin_files$To[i]
  print(paste("From", from_tree, "to", to_tree))
  
  print(paste("Loading", paste(moslin_data_path, moslin_files$Moslin_filename[i], sep = "")))
  edgelist_from <- tree_list_in[[from_tree]]$Edge_list
  edgelist_from <- edgelist_from[edgelist_from$Cell.type != "NA", ]
  edgelist_from$Cell <- sapply(edgelist_from$to, function(x){substr(x, 3, nchar(x))})
  colnames(edgelist_from)[1] <- "Node"
  edgelist_to <- tree_list_in[[to_tree]]$Edge_list
  edgelist_to <- edgelist_to[edgelist_to$Cell.type != "NA", ]
  edgelist_to$Cell <- sapply(edgelist_to$to, function(x){substr(x, 3, nchar(x))})
  colnames(edgelist_to)[1] <- "Node"

  transitions.moslin <-
    ReadTransitions(filename = paste(moslin_data_path, moslin_files$Moslin_filename[i], sep = ""),
                    edgelist_from = edgelist_from, edgelist_to = edgelist_to, sep = ",")
  names(transitions.moslin)[which(names(transitions.moslin) == "Probability")] <- "Prob_moslin"

  cell_types_t1 <- transitions.moslin[, .(Freq = uniqueN(From)), by = c("Cell_type_from")]
  cell_types_t1$Rel_freq <- cell_types_t1$Freq/sum(cell_types_t1$Freq)
  cell_types_t2 <- transitions.moslin[, .(Freq = uniqueN(To)), by = c("Cell_type_to")]
  cell_types_t2$Rel_freq <- cell_types_t2$Freq/sum(cell_types_t2$Freq)
  shared_types_over_cutoff <- intersect(cell_types_t1$Cell_type_from[cell_types_t1$Freq > freq_cutoff],
                                        cell_types_t2$Cell_type_to[cell_types_t1$Freq > freq_cutoff])
  off_type_probabilities <- transitions.moslin[Cell_type_from %in% shared_types_over_cutoff &
                                                 Cell_type_to %in% shared_types_over_cutoff]
  off_type_probabilities <- off_type_probabilities[Cell_type_from != Cell_type_to]
  
  inout_this <- 
    transitions.moslin[, .(Transfer_ratio = sum(Prob_moslin),
                           Cell_combinations = .N), by = c("Cell_type_from", "Cell_type_to")]
  if(nrow(off_type_probabilities) > 0){
    inout_this$Background_ratio <-
      sapply(inout_this$Cell_combinations,
             function(x){
               sum(off_type_probabilities$Prob_moslin[sample.int(n = nrow(off_type_probabilities),
                                                                 size = x, replace = T)])
             })
  }else{
    inout_this$Background_ratio <- NA
  }
  
  inout_this$Tree_from <- from_tree
  inout_this$Tree_to <- to_tree
  inout_this <- merge(inout_this, cell_types_t1, by = "Cell_type_from")
  colnames(inout_this)[c(ncol(inout_this) - 1, ncol(inout_this))] <- c("ct_from_freq", "ct_from_rel_freq")
  inout_this <- merge(inout_this, cell_types_t2, by = "Cell_type_to")
  colnames(inout_this)[c(ncol(inout_this) - 1, ncol(inout_this))] <- c("ct_to_freq", "ct_to_rel_freq")
  
  if(exists("transfer_ratios")){
    transfer_ratios <- rbind(transfer_ratios, inout_this)
  }else{
    transfer_ratios <- inout_this
  }
}
transfer_ratios_trees <- transfer_ratios
# fwrite(transfer_ratios, "./Data/All_celltype_transfer_ratios_with_background_tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9")
transfer_ratios_trees <-
  fread("../../../Data/All_celltype_transfer_ratios_with_background_tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9")

ct_and_transfer_averages <- AverageFrequenciesAndTransferRatios(transfer_ratios = transfer_ratios_trees,
                                                     tree_times = tree_time)
transfer_averages <- ct_and_transfer_averages$Transfer_ratios
ct_averages <- ct_and_transfer_averages$Cell_type_frequencies

background_transfer_ratios <- copy(transfer_ratios_trees)
background_transfer_ratios <- background_transfer_ratios[, Transfer_ratio:=NULL]
setnames(background_transfer_ratios, "Background_ratio", "Transfer_ratio")
background_ct_and_transfer_averages <- 
  AverageFrequenciesAndTransferRatios(transfer_ratios = background_transfer_ratios,
                                      tree_times = tree_time)

background_transfer_averages <- background_ct_and_transfer_averages$Transfer_ratios
setnames(background_transfer_averages, "Transfer_ratio", "Background_ratio")
transfer_with_background <- merge(transfer_averages, background_transfer_averages)

# Calculate percentages transferred to and from
transfer_averages_percentages <- copy(transfer_with_background)
transfer_averages_percentages <- transfer_averages_percentages[, Full_from := sum(Transfer_ratio), by = c("Cell_type_from", "t1_time")]
transfer_averages_percentages <- transfer_averages_percentages[, Full_to := sum(Transfer_ratio), by = c("Cell_type_to", "t2_time")]
transfer_averages_percentages$Perc_from <- 100 * transfer_averages_percentages$Transfer_ratio/transfer_averages_percentages$Full_from
transfer_averages_percentages$Perc_from[is.na(transfer_averages_percentages$Perc_from)] <- 0
transfer_averages_percentages$Perc_to <- 100 * transfer_averages_percentages$Transfer_ratio/transfer_averages_percentages$Full_to
transfer_averages_percentages$Perc_to[is.na(transfer_averages_percentages$Perc_to)] <- 0

# Identify largest percentages to and from that are not same celltype
cross_transfer_percentages <- 
  transfer_averages_percentages[transfer_averages_percentages$Cell_type_from != transfer_averages_percentages$Cell_type_to, ]

t1t2t3_alluvia <- CreateAlluvialDT(transfer_averages = ct_and_transfer_averages$Transfer_ratios,
                                   ct_averages = ct_and_transfer_averages$Cell_type_frequencies)

t1t2t3_alluvia_plot <- 
  CreateAlluvialPlotDT(alluvia_dt = t1t2t3_alluvia,
                       ct_averages = ct_averages,
                       fraction_threshold = 0.001)
# Figure 4d
ggplot(t1t2t3_alluvia_plot,
       aes(x = Time, stratum = Cell_type, alluvium = Alluvium,
           y = Fraction,
           fill = Cell_type, label = Cell_type)) +
  scale_x_continuous(expand = c(0, 0), breaks = 1:3, labels = c("Ctrl", "3dpi", "7dpi")) +
  scale_y_continuous(expand = c(0, 0), limits = c(-0.01, 1.03)) +
  labs(x = "", y = "Cell type fraction") + #, title = "Unbalanced cell type flows") +
  geom_flow(aes(alpha = I(Plot_alpha))) +
  geom_stratum(size = 0) +#alpha = .5) +
  scale_fill_manual(values = celltype_colors) +
  theme(legend.position = "none",
        text = element_text(family = "Helvetica Neue"),
        axis.title.y = element_text(size = 7),
        axis.title.x = element_blank(),
        axis.text = element_text(size = 6),
        panel.background = element_blank())
# ggsave("./Images/Alluvium_article_tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9.png",
# width = 3.5, height = 1.8, units = "in", device = png, dpi = 900, type = "cairo")

# Rules for zooming in:
# List a cell type and timepoint, a primary cutoff (lower) and a secondary cutoff (higher).
# All transitions to and from the cell type that are under the primary cutoff are set to NA.
# All transitions to or from the non-NA cell types at other timepoints that are under the
# secondary cutoff are set to NA.
ct_of_interest <- c("Fibroblasts (const.)", "Fibroblasts (col11a1a)", "Fibroblasts (col12a1a)")
ct_and_transfer_averages <- 
  AverageFrequenciesAndTransferRatios(transfer_ratios = transfer_ratios_trees,
                                      tree_times = unique(tree_time[, c("Tree", "time")]))
ct_and_transfer_zoom_averages <- 
  ZoomAverages(ct_and_transfer_averages, ct_of_interest)

transfer_averages_zoom <- ct_and_transfer_zoom_averages$Transfer_ratios
ct_averages_zoom <- ct_and_transfer_zoom_averages$Cell_type_frequencies
t1t2t3_alluvia_zoom <- CreateAlluvialDT(transfer_averages = transfer_averages_zoom,
                                        ct_averages = ct_averages_zoom)
t1t2t3_alluvia_zoom_plot <- CreateAlluvialPlotDT(alluvia_dt = t1t2t3_alluvia_zoom,
                                                 ct_averages = ct_averages_zoom,
                                                 fraction_threshold = 0)
t1t2t3_alluvia_zoom_plot[Cell_type == "Other"]$Fraction <- 0

# Remove flows by setting the alpha on the outgoing timepoint to 0.
t1t2t3_alluvia_zoom_plot$Show_alluvium <-
  apply(t1t2t3_alluvia_zoom_plot[, c("Alluvium", "Time")], 1,
      function(x){
        if(x[2] == 3){
          return(T)
        }else{
          unlist(strsplit(as.character(x[1]), "_"))[as.integer(x[2])] %in% ct_of_interest
          }})
t1t2t3_alluvia_zoom_plot$Plot_alpha[!t1t2t3_alluvia_zoom_plot$Show_alluvium] <- 0
# Plot 4e
ggplot(t1t2t3_alluvia_zoom_plot,
       aes(x = Time, stratum = Cell_type, alluvium = Alluvium,
           y = Fraction,
           fill = Cell_type, label = Cell_type)) +
  scale_x_continuous(expand = c(0, 0), breaks = 1:3, labels = c("Ctrl", "3dpi", "7dpi")) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 0.11)) +
  labs(x = "", y = "Cell type fraction")+#, title = "Unbalanced cell type flows") +
  geom_flow(aes(alpha = I(Plot_alpha))) +
  geom_stratum(size = 0) +#alpha = .5) +
  scale_fill_manual(values = celltype_colors) +
  theme(legend.position = "none",
        text = element_text(family = "Helvetica Neue"),
        axis.title.y = element_text(size = 7),
        axis.title.x = element_blank(),
        axis.text = element_text(size = 6),
        panel.background = element_blank())
# ggsave("./Images/Alluvium_article_tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9_col1112const_fixed_t3.png",
#        width = 2.7, height = 1.4, units = "in", device = png, dpi = 900, type = "cairo")

# Bootstrap for percentage confidence intervals
# Create all existing combinations: pick ctrl, 3dpi, 7dpi datasets (3(4), 6(9), 5(7) for a 50% + 48% downsampling, 
# total possible space of 7056 combinations).
ct_tofrom <- unique(paste(transfer_ratios_trees$Cell_type_from, 
                          transfer_ratios_trees$Cell_type_to, sep = "_"))
trees_tofrom <- unique(paste(transfer_ratios_trees$Tree_from, transfer_ratios_trees$Tree_to, sep = "_"))
all_ct_combinations <-
  data.table(expand.grid(ct_tf = ct_tofrom,
                         trees_tf = trees_tofrom, stringsAsFactors = F))
all_ct_combinations <-
  all_ct_combinations[, {
    cell_types <- unlist(strsplit(ct_tf, "_"))
    trees <- unlist(strsplit(trees_tf, "_"))
    list(Cell_type_from = cell_types[1],
         Cell_type_to = cell_types[2],
         Tree_from = trees[1],
         Tree_to = trees[2])
  }, by = names(all_ct_combinations)]

bootstrapped_average_percentages <- 
  transfer_averages_percentages[, c("t1_time", "t2_time", "Cell_type_from", "Cell_type_to")]
bsamples_ctrl <- 3
bsamples_3dpi <- 6
bsamples_7dpi <- 5
trees_ctrl <- unique(tree_time[time == "Ctrl"]$Tree)
trees_3dpi <- unique(tree_time[time == "3dpi"]$Tree)
trees_7dpi <- unique(tree_time[time == "7dpi"]$Tree)
combs_ctrl <- combn(trees_ctrl, bsamples_ctrl)
combs_3dpi <- combn(trees_3dpi, bsamples_3dpi)
combs_7dpi <- combn(trees_7dpi, bsamples_7dpi)
x <- expand.grid(combs_ctrl = 1:ncol(combs_ctrl), combs_3dpi = 1:ncol(combs_3dpi), combs_7dpi = 1:ncol(combs_7dpi))
combs_all <- matrix(data = "foo", nrow = bsamples_ctrl + bsamples_3dpi + bsamples_7dpi, ncol = nrow(x))

for(i in 1:nrow(x)){
  print(paste(i, nrow(x), sep = "/"))
  this_combi_trees <- c(combs_ctrl[, x[i, 1]], combs_3dpi[, x[i, 2]], combs_7dpi[, x[i, 3]])
  combs_all[, i] <- c(combs_ctrl[, x[i, 1]], combs_3dpi[, x[i, 2]], combs_7dpi[, x[i, 3]])
  # Filter transfer_ratios_trees accordingly.
  transfer_ratios_trees_bsample <- transfer_ratios_trees[Tree_from %in% this_combi_trees &
                                                           Tree_to %in% this_combi_trees]
  # Calculate averages: ct_and_transfer_averages <- AverageFrequenciesAndTransferRatios(transfer_ratios = transfer_ratios_trees, tree_times = unique(ct_freqs[, c("Tree", "time")]))
  ct_and_transfer_averages_bsample <- 
    AverageFrequenciesAndTransferRatios(transfer_ratios = transfer_ratios_trees_bsample,
                                        tree_times = unique(tree_time[Tree %in% this_combi_trees, c("Tree", "time")]),
                                        all_ct_combinations = all_ct_combinations)
  # Calculate percentages
  transfer_averages_bsample <- ct_and_transfer_averages_bsample$Transfer_ratios
  transfer_averages_bsample_percentages <- transfer_averages_bsample[, Full_from := sum(Transfer_ratio), by = c("Cell_type_from", "t1_time")]
  transfer_averages_bsample_percentages <- transfer_averages_bsample_percentages[, Full_to := sum(Transfer_ratio), by = c("Cell_type_to", "t2_time")]
  transfer_averages_bsample_percentages$Perc_from <- 100 * transfer_averages_bsample_percentages$Transfer_ratio/transfer_averages_bsample_percentages$Full_from
  transfer_averages_bsample_percentages$Perc_from[is.na(transfer_averages_bsample_percentages$Perc_from)] <- 0
  transfer_averages_bsample_percentages$Perc_to <- 100 * transfer_averages_bsample_percentages$Transfer_ratio/transfer_averages_bsample_percentages$Full_to
  transfer_averages_bsample_percentages$Perc_to[is.na(transfer_averages_bsample_percentages$Perc_to)] <- 0
  # Log percentages
  bootstrapped_average_percentages <- 
    merge(
      bootstrapped_average_percentages, 
      transfer_averages_bsample_percentages[, c("t1_time", "t2_time", "Cell_type_from", "Cell_type_to", 
                                                "Perc_from", "Perc_to")],
      all = T)
  colnames(bootstrapped_average_percentages)[(ncol(bootstrapped_average_percentages) -1 ):(ncol(bootstrapped_average_percentages))] <-
    c(paste("Perc_from", i, sep = "_"), paste("Perc_to", i, sep = "_"))
  }

# Calculate confidence intervals

# fwrite(bootstrapped_average_percentages,
#        "./Data/Bootstrapped_average_percentages_3-6-5_tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9")
# bootstrapped_average_percentages <-
#   fread("./Data/Bootstrapped_average_percentages_3-6-5_unbalanced_alpha-0.5_epsilon-0.001_beta-0.2_taua-0.8")

names(bootstrapped_average_percentages)
bap_long <- melt(bootstrapped_average_percentages, 
                 id.vars = c("t1_time", "t2_time", "Cell_type_from", "Cell_type_to"),
                 variable.name = "Sample_type", value.name = "Percentage", variable.factor = F)
bap_long <-
  bap_long[, {
    x <- unlist(strsplit(Sample_type, "_"))
    list(Type = x[2],
         Sample = x[3])}, by = names(bap_long)]
bap_ci <- bap_long[, {
  perc_quantile = quantile(Percentage, probs = seq(0, 1, 0.025), na.rm = T)
  list(Min_095 = perc_quantile[2],
       Max_095 = perc_quantile[40])},
  by = c("t1_time", "t2_time", "Cell_type_from", "Cell_type_to", "Type")]

tap_ci <- 
  merge(transfer_averages_percentages, 
        bap_ci[Type == "from", c("t1_time", "t2_time", "Cell_type_from", "Cell_type_to", "Min_095", "Max_095")],
        all = T)
colnames(tap_ci)[(ncol(tap_ci) - 1):ncol(tap_ci)] <- c("Perc_from_min_095", "Perc_from_max_095")
tap_ci <- 
  merge(tap_ci, 
        bap_ci[Type == "to", c("t1_time", "t2_time", "Cell_type_from", "Cell_type_to", "Min_095", "Max_095")],
        all = T)
colnames(tap_ci)[(ncol(tap_ci) - 1):ncol(tap_ci)] <- c("Perc_to_min_095", "Perc_to_max_095")
# fwrite(tap_ci, "./Data/Bootstrapped_95_ci_3-6-5_tmats_moslin_alpha-0.5_epsilon-0.01_beta-0.2_taua-0.9")
# tap_ci <- fread("./Data/Bootstrapped_95_ci_3-6-5_unbalanced_alpha-0.5_epsilon-0.001_beta-0.2_taua-0.8")
