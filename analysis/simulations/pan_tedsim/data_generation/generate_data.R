library(TedSim)
library(dplyr)
library(reshape2)
library(ctc)
library(ape)
library(dichromat)
library(scales)

args <- commandArgs(trailingOnly=T)
p_a <- as.numeric(args[1])
ss <- as.numeric(args[2])
seed  <- as.numeric(args[3])

ncells <- 8192  # 2 ^ 13
phyla <- read.tree(text='((t1:2, t2:2):1, (t3:2, t4:2):1):2;')
N_nodes <- 2 * ncells-2
ngenes <- 500
max_walk <- 6
n_cif <- 30
n_diff <- 20
p_d <- 0
mu <- 0.1
N_char <- 32
unif_on <- FALSE

generate <- function(p_a, cif_step, seed) {
        modifier <- paste("", p_a, cif_step, seed, sep='_')
        set.seed(seed)

        returnlist <- SIFGenerate(phyla, n_diff, step = cif_step)
        cifs <- SimulateCIFs(ncells,phyla,p_a = p_a,n_CIF = n_cif,n_diff = n_diff,
                     step = cif_step, p_d = p_d, mu = mu,
                     Sigma = 0.5, N_char = N_char,
                     max_walk = max_walk, SIF_res = returnlist, unif_on = unif_on)
        # we only need the leaf cells for experiments
        cif_leaves <- lapply(c(1:3),function(parami){
          cif_leaves_all <- cifs[[1]][[parami]][c(1:ncells),]
          return(cif_leaves_all)
        })
        cif_res <- list(cif_leaves,cifs[[2]])
        states <- cifs[[2]]
        states <- states[1:N_nodes,]
        muts <- cifs[[7]]
        rownames(muts) <- paste("cell",states[,4],sep = "_")

        # simulate true counts
        true_counts_res <- CIF2Truecounts(ngenes = ngenes, ncif = n_cif,
                                          ge_prob = 0.3,ncells = N_nodes, cif_res = cifs)

        data(gene_len_pool)
        gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
        observed_counts <- True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[3]],
                                               protocol="UMI", alpha_mean=0.2,
                                               alpha_sd=0.05, gene_len=gene_len, depth_mean=1e5, depth_sd=3e3)

        gene_expression_dir <- paste0("./output/counts_tedsim", modifier, ".csv")
        cell_meta_dir <- paste0("./output/cell_meta_tedsim", modifier, ".csv")
        character_matrix_dir <- paste0("./output/character_matrix", modifier, ".txt")
        tree_gt_dir <- paste0("./output/tree_gt_bin_tedsim", modifier, ".newick")

        write.tree(cifs[[4]], tree_gt_dir)
        write.csv(observed_counts[[1]], gene_expression_dir, row.names=FALSE)
        write.csv(states, cell_meta_dir)
        write.table(muts, character_matrix_dir)

        write.csv(cifs[[4]]$edge, paste0("./output/edges", modifier, ".csv"))
        write.csv(states, paste0("./output/states_full", modifier, ".csv"))
}


generate(p_a, ss, seed)
