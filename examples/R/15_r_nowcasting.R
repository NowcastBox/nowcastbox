# R counterpart of notebook 15: the examples of ?nowcast (R package `nowcasting`).
#
# The GPL-3 package `nowcasting` (de Valk, de Mattos & Ferreira, 2019) is used here only as
# a BLACK BOX: this script calls its documented functions, exactly as in the examples of
# its help pages, and stores the outputs and run times. nowcastbox (MIT) never reads or
# translates its source code.
#
# Usage (from the repository root):
#   Rscript examples/R/15_r_nowcasting.R
# Writes examples/outputs/R/{r_timings.csv, usgdp_2s_agg_yfcst.csv, nyfed_em_yfcst.csv}.
# Notebook 15 reads r_timings.csv when it exists.

suppressMessages(library(nowcasting))

out_dir <- file.path("examples", "outputs", "R")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

timed <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, seconds = proc.time()[["elapsed"]] - start)
}

write_yfcst <- function(yfcst, file) {
  periods <- sprintf("%dQ%d", floor(time(yfcst) + 1e-8), cycle(yfcst))
  frame <- data.frame(period = periods, as.data.frame(unclass(yfcst)))
  write.csv(frame, file.path(out_dir, file), row.names = FALSE)
}

# --- Two-step DFM with factor aggregation (Giannone, Reichlin & Small, 2008): USGDP -----
data(USGDP)
usgdp <- Bpanel(base = USGDP$base, trans = USGDP$legend$Transformation, aggregate = FALSE)
frequency <- c(rep(12, ncol(usgdp) - 1), 4)
two_step <- timed(nowcast(formula = RGDPGR ~ ., data = usgdp, r = 2, p = 2, q = 2,
                          method = "2s_agg", frequency = frequency))
write_yfcst(two_step$value$yfcst, "usgdp_2s_agg_yfcst.csv")

# --- EM with blocks (Banbura & Modugno, 2014): the NY Fed example ------------------------
data(NYFED)
nyfed <- Bpanel(base = NYFED$base, trans = NYFED$legend$Transformation,
                NA.replace = FALSE, na.prop = 1)
em <- timed(nowcast(formula = GDPC1 ~ ., data = nyfed, r = 1, p = 1, method = "EM",
                    blocks = NYFED$blocks$blocks, frequency = NYFED$legend$Frequency))
write_yfcst(em$value$yfcst, "nyfed_em_yfcst.csv")

timings <- data.frame(
  case = c("usgdp_2s_agg", "nyfed_em"),
  seconds = c(two_step$seconds, em$seconds),
  r_version = R.version.string,
  nowcasting_version = as.character(packageVersion("nowcasting"))
)
write.csv(timings, file.path(out_dir, "r_timings.csv"), row.names = FALSE)
print(timings)
