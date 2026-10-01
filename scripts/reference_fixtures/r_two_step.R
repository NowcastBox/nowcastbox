# Reference outputs of nowcast(method = "2s" / "2s_agg") of the R package "nowcasting"
# (black box) on USGDP (the package example, Giannone, Reichlin & Small, 2008) and on the
# shared simulated panel.
#
# Run from the repository root:  Rscript scripts/reference_fixtures/r_two_step.R

source(file.path("scripts", "reference_fixtures", "_common.R"))

quiet <- function(expr) suppressWarnings(suppressMessages(expr))
timings <- list()

export_two_step <- function(fit, prefix, r, p, q, panel_columns, x_columns = NULL) {
  f <- fit$factors
  write_ts_csv(f$dynamic_factors, file.path(DIR_R, paste0(prefix, "_factors.csv")))
  write_ts_csv(fit$yfcst, file.path(DIR_R, paste0(prefix, "_yfcst.csv")))
  if (!is.null(x_columns)) {
    write_ts_csv(fit$xfcst[, x_columns, drop = FALSE],
                 file.path(DIR_R, paste0(prefix, "_xfcst.csv.gz")))
  }
  lambda <- f$Lambda[, seq_len(r), drop = FALSE]
  write_json_fixture(list(
    r = r, p = p, q = q,
    columns = panel_columns,
    A = f$A, BB = f$BB, Lambda = lambda, Psi = diag(f$Psi),
    initx = as.numeric(f$initx), initV = f$initV,
    eigenvalues = as.numeric(f$eigen$values),
    mean = as.numeric(f$mean), std = as.numeric(f$std),
    reg_coefficients = as.list(coef(fit$reg)),
    reg_sigma = summary(fit$reg)$sigma,
    reg_nobs = length(fit$reg$residuals)
  ), paste0(prefix, ".json"))
}

# --- USGDP, method "2s" (the package example) ---------------------------------------
data(USGDP)
gdp_position <- which(colnames(USGDP$base) == "RGDPGR")
x_agg <- quiet(Bpanel(base = USGDP$base[, -gdp_position],
                      trans = USGDP$legend$Transformation[-gdp_position], aggregate = TRUE))
write_ts_csv(column_subset(x_agg), file.path(DIR_R, "usgdp_2s_panel_subset.csv.gz"))
data_2s <- cbind(USGDP$base[, "RGDPGR"], x_agg)
colnames(data_2s) <- c("RGDPGR", colnames(x_agg))
freq_2s <- c(4, rep(12, ncol(data_2s) - 1))
run <- timed(quiet(nowcast(formula = RGDPGR ~ ., data = data_2s, r = 2, p = 2, q = 2,
                           method = "2s", frequency = freq_2s)), 3)
timings$usgdp_2s <- run$seconds
export_two_step(run$value, "usgdp_2s", 2, 2, 2, colnames(x_agg),
                x_columns = colnames(x_agg)[1:5])

# --- USGDP, method "2s_agg" (the package example) -----------------------------------
data_agg <- quiet(Bpanel(base = USGDP$base, trans = USGDP$legend$Transformation,
                         aggregate = FALSE))
write_ts_csv(column_subset(data_agg), file.path(DIR_R, "usgdp_2s_agg_panel_subset.csv.gz"))
freq_agg <- ifelse(colnames(data_agg) == "RGDPGR", 4, 12)
run <- timed(quiet(nowcast(formula = RGDPGR ~ ., data = data_agg, r = 2, p = 2, q = 2,
                           method = "2s_agg", frequency = freq_agg)), 3)
timings$usgdp_2s_agg <- run$seconds
x_names <- setdiff(colnames(data_agg), "RGDPGR")
export_two_step(run$value, "usgdp_2s_agg", 2, 2, 2, x_names, x_columns = x_names[1:5])

# --- simulated panel ----------------------------------------------------------------
# nowcast() needs empty rows for the forecast horizon (Bpanel adds h = 12 of them).
sim <- pad_rows(read_monthly_csv(file.path(DIR_INPUTS, "simulated.csv")), 12)
x_names <- setdiff(colnames(sim), "gdp")
freq_sim <- ifelse(colnames(sim) == "gdp", 4, 12)
run <- timed(quiet(nowcast(formula = gdp ~ ., data = sim, r = 2, p = 1, q = 2,
                           method = "2s_agg", frequency = freq_sim)), 5)
timings$simulated_2s_agg <- run$seconds
export_two_step(run$value, "sim_2s_agg", 2, 1, 2, x_names, x_columns = x_names)

run <- timed(quiet(nowcast(formula = gdp ~ ., data = sim, r = 2, p = 1, q = 1,
                           method = "2s_agg", frequency = freq_sim)), 5)
export_two_step(run$value, "sim_2s_agg_q1", 2, 1, 1, x_names)

# "2s": the predictors are filtered with Bpanel(aggregate = TRUE) (no transformation;
# the 4 leading periods lost to the filter are filled by Bpanel - nowcast(method = "2s")
# fails on leading missing values - and the Gaussian panel has no outlier beyond 4 IQR).
x_sim_agg <- quiet(Bpanel(sim[, x_names], rep(0, length(x_names)), aggregate = TRUE,
                          na.prop = 1, h = 0))
data_sim_2s <- cbind(sim[, "gdp"], x_sim_agg)
colnames(data_sim_2s) <- c("gdp", colnames(x_sim_agg))
run <- timed(quiet(nowcast(formula = gdp ~ ., data = data_sim_2s, r = 2, p = 1, q = 2,
                           method = "2s", frequency = c(4, rep(12, ncol(x_sim_agg))))), 5)
timings$simulated_2s <- run$seconds
export_two_step(run$value, "sim_2s", 2, 1, 2, colnames(x_sim_agg))
write_ts_csv(x_sim_agg, file.path(DIR_R, "sim_2s_panel.csv.gz"))

# Balanced variant (no ragged edge in the predictors): R standardises over the balanced
# rows and nowcastbox over every observed value; here both samples coincide.
sim_bal <- pad_rows(window(read_monthly_csv(file.path(DIR_INPUTS, "simulated.csv")),
                           end = c(2019, 9)), 12)
run <- timed(quiet(nowcast(formula = gdp ~ ., data = sim_bal, r = 2, p = 1, q = 2,
                           method = "2s_agg", frequency = freq_sim)), 5)
export_two_step(run$value, "simbal_2s_agg", 2, 1, 2, x_names, x_columns = x_names[1:3])
run <- timed(quiet(nowcast(formula = gdp ~ ., data = sim_bal, r = 2, p = 2, q = 2,
                           method = "2s_agg", frequency = freq_sim)), 5)
export_two_step(run$value, "simbal_2s_agg_p2", 2, 2, 2, x_names)

record_timings(timings)
message("two-step fixtures written to ", DIR_R)
