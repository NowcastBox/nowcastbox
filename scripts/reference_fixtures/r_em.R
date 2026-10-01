# Reference outputs of nowcast(method = "EM") of the R package "nowcasting" (black box)
# on NYFED (the package example: replication of the FRBNY Staff Nowcast, 4 blocks with
# one factor each) and on the shared simulated panel (one block, two factors).
#
# The log-likelihood is not returned by nowcast(); the values the function prints every
# five iterations are captured from its console output. The final parameters
# (Res$A, Res$C, Res$Q, Res$R, Res$Z_0, Res$V_0, Res$Mx, Res$Wx) are exported so that
# the tests can evaluate the R model with the nowcastbox Kalman filter.
#
# Run from the repository root:  Rscript scripts/reference_fixtures/r_em.R   (~1-2 min)

source(file.path("scripts", "reference_fixtures", "_common.R"))

quiet <- function(expr) suppressWarnings(suppressMessages(expr))
timings <- list()

# Run an expression, returning its value, elapsed time and the console lines it printed
# (nowcast() reports the EM log-likelihood through message()).
run_logged <- function(expr) {
  f <- substitute(expr)
  env <- parent.frame()
  msgs <- character()
  t0 <- proc.time()[["elapsed"]]
  out <- utils::capture.output(
    value <- withCallingHandlers(
      suppressWarnings(eval(f, env)),
      message = function(m) {
        msgs <<- c(msgs, conditionMessage(m))
        invokeRestart("muffleMessage")
      }
    )
  )
  list(value = value, seconds = proc.time()[["elapsed"]] - t0,
       lines = unlist(strsplit(c(out, msgs), "\n")))
}

parse_loglik <- function(lines) {
  hits <- grep("loglikelihood went from", lines, value = TRUE)
  iters <- grep("iteration", lines, value = TRUE)
  it <- as.integer(sub("^([0-9]+).*$", "\\1", iters))
  from <- as.numeric(sub(".*from ([-0-9.e]+) to.*", "\\1", hits))
  to <- as.numeric(sub(".* to ([-0-9.e]+).*$", "\\1", hits))
  list(iteration = it, from = from, to = to)
}

export_em <- function(fit, prefix, columns, blocks, r, p, loglik, factor_columns) {
  res <- fit$Res
  # nowcast() reorders the series internally (monthly first): label Res by its own names
  internal <- names(res$Mx)
  if (is.null(internal)) internal <- columns
  write_ts_csv(fit$yfcst, file.path(DIR_R, paste0(prefix, "_yfcst.csv")))
  ff <- fit$factors$dynamic_factors
  write_ts_csv(ff, file.path(DIR_R, paste0(prefix, "_factors.csv")))
  x_sm <- ts(res$x_sm, start = start(ff), frequency = 12)
  colnames(x_sm) <- internal
  write_ts_csv(x_sm, file.path(DIR_R, paste0(prefix, "_x_sm.csv.gz")))
  write_json_fixture(list(
    columns = internal, input_columns = columns, blocks = blocks, r = r, p = p,
    factor_columns = factor_columns,
    A = res$A, C = res$C, Q = res$Q, R = diag(res$R),
    Z_0 = as.numeric(res$Z_0), V_0 = res$V_0,
    Mx = as.numeric(res$Mx), Wx = as.numeric(res$Wx),
    loglik_path = loglik
  ), paste0(prefix, ".json"))
}

# --- NYFED ----------------------------------------------------------------------------
data(NYFED)
blocks <- NYFED$blocks$blocks
trans <- NYFED$legend$Transformation
frequency <- NYFED$legend$Frequency
panel <- quiet(Bpanel(base = NYFED$base, trans = trans, NA.replace = FALSE, na.prop = 1))
write_ts_csv(panel, file.path(DIR_R, "nyfed_em_panel.csv.gz"), digits = 15)
run <- run_logged(nowcast(formula = GDPC1 ~ ., data = panel, r = 1, p = 1, method = "EM",
                          blocks = blocks, frequency = frequency))
timings$nyfed_em <- run$seconds
export_em(run$value, "nyfed_em", colnames(panel), blocks, 1, 1, parse_loglik(run$lines),
          factor_columns = colnames(run$value$factors$dynamic_factors))

# --- simulated panel: one global block, two factors ----------------------------------
sim <- pad_rows(read_monthly_csv(file.path(DIR_INPUTS, "simulated.csv")), 12)
freq_sim <- ifelse(colnames(sim) == "gdp", 4, 12)
blocks_sim <- matrix(1, ncol(sim), 1)
run <- run_logged(nowcast(formula = gdp ~ ., data = sim, r = 2, p = 1, method = "EM",
                          blocks = blocks_sim, frequency = freq_sim))
timings$simulated_em <- run$seconds
export_em(run$value, "sim_em", colnames(sim), blocks_sim, 2, 1, parse_loglik(run$lines),
          factor_columns = colnames(run$value$factors$dynamic_factors))

record_timings(timings)
message("EM fixtures written to ", DIR_R)
