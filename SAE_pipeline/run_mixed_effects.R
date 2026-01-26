#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(dplyr)
    library(stats)
    # We try to load glmmTMB, but won't crash if it's broken until we use it
    try(library(glmmTMB), silent = TRUE)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
    stop("Usage: Rscript run_mixed_effects.R <input_dir> <output_dir>")
}

input_dir <- args[1]
output_dir <- args[2]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Helper Functions
# ==============================================================================

# Clamp values to (epsilon, 1-epsilon) for Beta/Logit logic
clamp_beta <- function(x, eps = 1e-5) {
    pmin(pmax(x, eps), 1 - eps)
}

# Check if a column has >1 unique value
has_variance <- function(df, col) {
    if (!col %in% names(df)) return(FALSE)
    return(length(unique(df[[col]])) > 1)
}

# Filter data to require at least min_samples per context category
filter_by_context_sample_size <- function(df, context_col = "context", min_samples = 3, model_name = "Model") {
    if (!context_col %in% names(df)) {
        cat(paste0("  [INFO] No '", context_col, "' column found, skipping sample size filter.\n"))
        return(df)
    }
    
    # Count samples per context - use n_occurrences if available, otherwise count rows
    if ("n_occurrences" %in% names(df)) {
        # Sum total occurrences per context
        context_counts <- df %>%
            group_by(.data[[context_col]]) %>%
            summarise(n = sum(n_occurrences, na.rm = TRUE), .groups = "drop")
        count_type <- "occurrences"
    } else {
        # Count rows per context
        context_counts <- df %>%
            group_by(.data[[context_col]]) %>%
            summarise(n = n(), .groups = "drop")
        count_type <- "samples"
    }
    
    # Identify contexts with sufficient samples
    valid_contexts <- context_counts %>%
        filter(n >= min_samples) %>%
        pull(.data[[context_col]])
    
    # Report filtering
    filtered_contexts <- context_counts %>%
        filter(n < min_samples)
    
    if (nrow(filtered_contexts) > 0) {
        cat(paste0("  [FILTER] ", model_name, ": Filtering out contexts with < ", min_samples, " ", count_type, ":\n"))
        for (i in 1:nrow(filtered_contexts)) {
            cat(paste0("    - ", filtered_contexts[[context_col]][i], ": ", filtered_contexts$n[i], " ", count_type, "\n"))
        }
        cat(paste0("  [FILTER] Keeping ", length(valid_contexts), " contexts with >= ", min_samples, " ", count_type, "\n"))
    } else {
        cat(paste0("  [INFO] All ", length(valid_contexts), " contexts have >= ", min_samples, " ", count_type, "\n"))
    }
    
    # Filter data
    df_filtered <- df %>%
        filter(.data[[context_col]] %in% valid_contexts)
    
    return(df_filtered)
}

build_formula <- function(df, target_var, use_random = TRUE) {
    # Fixed effects
    fixed <- c()
    if (has_variance(df, "polysemy")) fixed <- c(fixed, "polysemy")
    if (has_variance(df, "context")) fixed <- c(fixed, "context")
    if (has_variance(df, "layer")) fixed <- c(fixed, "layer")
    if (has_variance(df, "model")) fixed <- c(fixed, "model")
    
    # Construct RHS
    if (length(fixed) == 0) {
        rhs <- "1"
    } else {
        rhs <- paste(fixed, collapse = " + ")
    }
    
    # Random effects (Only for Mixed Models)
    if (use_random) {
        rhs <- paste(rhs, "+ (1 | lemma)")
        if (has_variance(df, "doc_id")) rhs <- paste(rhs, "+ (1 | doc_id)")
    }
    
    return(as.formula(paste(target_var, "~", rhs)))
}

fit_robust <- function(df, target_var, model_type = "beta", name = "Model") {
    cat(paste0("--- Fitting ", name, " (N=", nrow(df), ") ---\n"))
    
    if (nrow(df) < 5) {
        cat("  [SKIP] Too few data points.\n")
        return(NULL)
    }
    
    try_tmb <- tryCatch({
        f <- build_formula(df, target_var, use_random = TRUE)
        
        family_fn <- NULL
        if (model_type == "beta") family_fn <- beta_family(link = "logit")
        if (model_type == "gamma") family_fn <- Gamma(link = "log")
        
        cat("  Attempting glmmTMB:", deparse(f), "\n")
        m <- glmmTMB(f, data = df, family = family_fn)
        if (any(is.nan(logLik(m)))) stop("NaN Log-Likelihood")
        m
    }, error = function(e) {
        cat("  [FAIL] glmmTMB failed:", conditionMessage(e), "\n")
        return(NULL)
    })
    
    if (!is.null(try_tmb)) {
        cat("  [SUCCESS] glmmTMB converged.\n")
        saveRDS(try_tmb, file.path(output_dir, paste0("model_", tolower(gsub(" ", "_", name)), ".rds")))
        return(summary(try_tmb)$coefficients$cond)
    }
    
    cat("  [FALLBACK] Switching to Base R models (ignoring random effects)...\n")
    f_fixed <- build_formula(df, target_var, use_random = FALSE)
    
    try_base <- tryCatch({
        if (model_type == "beta") {
            y_trans <- paste0("qlogis(", target_var, ")")
            f_str <- deparse(f_fixed)
            f_str <- sub(paste0("^", target_var), y_trans, f_str)
            f_lm <- as.formula(f_str)
            
            cat("  Attempting LM (Logit):", f_str, "\n")
            m <- lm(f_lm, data = df)
            m
        } else if (model_type == "gamma") {
            cat("  Attempting GLM (Gamma):", deparse(f_fixed), "\n")
            m <- glm(f_fixed, data = df, family = Gamma(link = "log"))
            m
        }
    }, error = function(e) {
        cat("  [FAIL] Fallback failed:", conditionMessage(e), "\n")
        return(NULL)
    })
    
    if (!is.null(try_base)) {
        cat("  [SUCCESS] Fallback model converged.\n")
        # Normalize coefficient output format
        coefs <- summary(try_base)$coefficients
        return(coefs)
    }
    
    return(NULL)
}

# ==============================================================================
# Main Execution
# ==============================================================================

# --- Load & Preprocess ---

# 1. Entropy
path_ent <- file.path(input_dir, "entropy_per_sense.csv")
if (file.exists(path_ent)) {
    df_entropy <- read.csv(path_ent, stringsAsFactors = FALSE)
    if (nrow(df_entropy) > 0) {
        df_entropy <- df_entropy %>%
            group_by(lemma) %>%
            mutate(
                polysemy = ifelse(n_distinct(sense) > 1, "polysemantic", "monosemantic"),
                context = sense,
                entropy_scaled = clamp_beta((normalized_entropy - 0) / (1 - 0))
            ) %>% ungroup()
        
        df_entropy <- filter_by_context_sample_size(df_entropy, "context", min_samples = 3, "Entropy")
        
        if (nrow(df_entropy) > 0) {
            coefs <- fit_robust(df_entropy, "entropy_scaled", "beta", "Entropy")
            if (!is.null(coefs)) write.csv(coefs, file.path(output_dir, "coefs_entropy.csv"))
        } else {
            cat("  [SKIP] No data remaining after filtering contexts with < 3 samples\n")
        }
    }
} else {
    cat("Missing entropy_per_sense.csv\n")
}

# 2. Jaccard
path_jac <- file.path(input_dir, "jaccard_cross_sense.csv")
if (file.exists(path_jac)) {
    df_jaccard <- read.csv(path_jac, stringsAsFactors = FALSE)
    if (nrow(df_jaccard) > 0) {
        df_jaccard <- df_jaccard %>%
            mutate(
                jaccard_scaled = clamp_beta(mean_jaccard),
                polysemy = "polysemantic"
            )
        
        coefs <- fit_robust(df_jaccard, "jaccard_scaled", "beta", "Jaccard")
        if (!is.null(coefs)) write.csv(coefs, file.path(output_dir, "coefs_jaccard.csv"))
    }
} else {
    cat("Missing jaccard_cross_sense.csv\n")
}

# 3. Delta CE
path_dce <- file.path(input_dir, "ablation_delta_ce.csv")
if (file.exists(path_dce)) {
    df_delta_ce <- read.csv(path_dce, stringsAsFactors = FALSE)
    if (nrow(df_delta_ce) > 0) {
        df_delta_ce_pos <- df_delta_ce %>%
            filter(delta_ce > 0.001) %>%
            group_by(lemma) %>%
            mutate(
                polysemy = ifelse(n_distinct(sense) > 1, "polysemantic", "monosemantic"),
                context = sense
            ) %>% ungroup()
        
        df_delta_ce_pos <- filter_by_context_sample_size(df_delta_ce_pos, "context", min_samples = 3, "Delta CE")
        
        if (nrow(df_delta_ce_pos) > 0) {
            coefs <- fit_robust(df_delta_ce_pos, "delta_ce", "gamma", "Delta CE")
            if (!is.null(coefs)) write.csv(coefs, file.path(output_dir, "coefs_delta_ce.csv"))
        } else {
            cat("  [SKIP] No data remaining after filtering contexts with < 3 samples\n")
        }
    }
} else {
    cat("Missing ablation_delta_ce.csv\n")
}

# --- FDR Correction ---
cat("\n--- Applying FDR Correction ---\n")
all_pvals <- c()
all_terms <- c()

for (m in c("entropy", "jaccard", "delta_ce")) {
    fpath <- file.path(output_dir, paste0("coefs_", m, ".csv"))
    if (file.exists(fpath)) {
        res <- read.csv(fpath, row.names = 1)
        pval_col <- grep("Pr\\(", colnames(res), value = TRUE)
        if (length(pval_col) > 0) {
            p <- res[[pval_col[1]]]
            nm <- rownames(res)
            all_pvals <- c(all_pvals, p)
            all_terms <- c(all_terms, paste0(m, "__", nm))
        }
    }
}

if (length(all_pvals) > 0) {
    qvals <- p.adjust(all_pvals, method = "BH")
    df_fdr <- data.frame(term = all_terms, pvalue = all_pvals, qvalue = qvals)
    write.csv(df_fdr, file.path(output_dir, "fdr_correction.csv"), row.names = FALSE)
    
    cat("Significant terms (q < 0.05):\n")
    print(df_fdr[df_fdr$qvalue < 0.05, ])
} else {
    cat("No p-values found to correct.\n")
}

cat("Done.\n")