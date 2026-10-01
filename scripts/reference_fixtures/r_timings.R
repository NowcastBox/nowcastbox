# Wall-clock timings of the R package "nowcasting" (black box) for benchmarks/bench_vs_r.py.
#
# Times the same calls as the fixture scripts (median of `reps` runs, one R process,
# nothing else running) and merges them into tests/reference_validation/fixtures/r/
# timings.json. Run on an idle machine from the repository root (~5 minutes):
#
#   OMP_NUM_THREADS=1 Rscript scripts/reference_fixtures/r_timings.R

source(file.path("scripts", "reference_fixtures", "_common.R"))

quiet <- function(expr) suppressWarnings(suppressMessages(expr))
timings <- list()

data(USGDP)
data(NYFED)
data(BRGDP)

timings$bpanel_transform_brgdp <- timed(quiet(
  Bpanel(BRGDP$base, BRGDP$trans, NA.replace = FALSE, na.prop = 1, h = 0)), 5)$seconds
timings$bpanel_default_brgdp <- timed(quiet(Bpanel(BRGDP$base, BRGDP$trans, h = 0)), 5)$seconds
gdp_position <- which(colnames(USGDP$base) == "RGDPGR")
x_us <- USGDP$base[, -gdp_position]
tr_us <- USGDP$legend$Transformation[-gdp_position]
timings$bpanel_default_usgdp <- timed(quiet(Bpanel(x_us, tr_us)), 3)$seconds
x_agg <- quiet(Bpanel(x_us, tr_us, aggregate = TRUE))
timings$bpanel_aggregate_usgdp <- timed(quiet(Bpanel(x_us, tr_us, aggregate = TRUE)), 3)$seconds

data_2s <- cbind(USGDP$base[, "RGDPGR"], x_agg)
colnames(data_2s) <- c("RGDPGR", colnames(x_agg))
timings$usgdp_2s <- timed(quiet(nowcast(RGDPGR ~ ., data = data_2s, r = 2, p = 2, q = 2,
                                        method = "2s",
                                        frequency = c(4, rep(12, ncol(x_agg))))), 3)$seconds
data_agg <- quiet(Bpanel(USGDP$base, USGDP$legend$Transformation))
freq_agg <- ifelse(colnames(data_agg) == "RGDPGR", 4, 12)
timings$usgdp_2s_agg <- timed(quiet(nowcast(RGDPGR ~ ., data = data_agg, r = 2, p = 2, q = 2,
                                            method = "2s_agg", frequency = freq_agg)), 3)$seconds

sim <- pad_rows(read_monthly_csv(file.path(DIR_INPUTS, "simulated.csv")), 12)
freq_sim <- ifelse(colnames(sim) == "gdp", 4, 12)
timings$simulated_2s_agg <- timed(quiet(nowcast(gdp ~ ., data = sim, r = 2, p = 1, q = 2,
                                                method = "2s_agg", frequency = freq_sim)),
                                  5)$seconds
timings$simulated_em <- timed(quiet(nowcast(gdp ~ ., data = sim, r = 2, p = 1, method = "EM",
                                            blocks = matrix(1, ncol(sim), 1),
                                            frequency = freq_sim)), 3)$seconds

panel <- quiet(Bpanel(NYFED$base, NYFED$legend$Transformation, NA.replace = FALSE,
                      na.prop = 1))
timings$nyfed_em <- timed(quiet(nowcast(GDPC1 ~ ., data = panel, r = 1, p = 1, method = "EM",
                                        blocks = NYFED$blocks$blocks,
                                        frequency = NYFED$legend$Frequency)), 1)$seconds

us <- quiet(Bpanel(x_us, tr_us))
us <- us[stats::complete.cases(us), ]
timings$icfactors_usgdp <- timed(quiet(ICfactors(us, rmax = 15, type = 2)), 5)$seconds
timings$icshocks_usgdp <- timed(quiet(ICshocks(us, r = 4, p = 2)), 5)$seconds

vintages <- seq(as.Date("2008-01-15"), as.Date("2017-12-15"), by = "month")
t0 <- proc.time()[["elapsed"]]
for (v in as.character(vintages)) invisible(PRTDB(BRGDP$base, BRGDP$delay, vintage = v))
timings$prtdb_brgdp_per_vintage <- (proc.time()[["elapsed"]] - t0) / length(vintages)

record_timings(c(timings, list(timed_on = format(Sys.time(), "%Y-%m-%d %H:%M"))))
message("timings written to ", file.path(DIR_R, "timings.json"))
