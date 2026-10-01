# Reference outputs of PRTDB (pseudo real-time database) of the R package "nowcasting"
# (black box) on BRGDP and NYFED with their bundled publication delays.
#
# PRTDB only blanks values and truncates the rows after the vintage month, so a vintage
# is summarised per series by the number of values kept and the position of the last
# value kept (0-based row of the full input grid, -1 when nothing is kept). Daily vintages
# around month ends discriminate the release-date convention.
#
# Run from the repository root:  Rscript scripts/reference_fixtures/r_prtdb.R

source(file.path("scripts", "reference_fixtures", "_common.R"))

data(BRGDP)
data(NYFED)
timings <- list()

summarise_vintages <- function(base, delay, vintages, prefix) {
  n_obs <- matrix(NA_integer_, length(vintages), ncol(base))
  last <- matrix(NA_integer_, length(vintages), ncol(base))
  rows <- nrow(base)
  t0 <- proc.time()[["elapsed"]]
  for (k in seq_along(vintages)) {
    v <- PRTDB(mts = base, delay = delay, vintage = vintages[k])
    m <- as.matrix(v)
    kept <- !is.na(m)
    # PRTDB keeps the original values: check while we are here
    stopifnot(all(m[kept] == as.matrix(base)[seq_len(nrow(m)), , drop = FALSE][kept]))
    n_obs[k, ] <- colSums(kept)
    last[k, ] <- apply(kept, 2, function(col) if (any(col)) max(which(col)) - 1L else -1L)
  }
  elapsed <- proc.time()[["elapsed"]] - t0
  colnames(n_obs) <- colnames(base)
  colnames(last) <- colnames(base)
  for (what in c("n_obs", "last")) {
    mat <- if (what == "n_obs") n_obs else last
    df <- data.frame(vintage = as.character(vintages), mat, check.names = FALSE)
    con <- gzfile(file.path(DIR_R, sprintf("prtdb_%s_%s.csv.gz", prefix, what)), "w")
    write.csv(df, con, row.names = FALSE, quote = FALSE)
    close(con)
  }
  elapsed / length(vintages)
}

daily <- function(from, to) seq(as.Date(from), as.Date(to), by = "day")
monthly <- function(from, to) seq(as.Date(from), as.Date(to), by = "month")

v_br <- sort(unique(c(daily("2017-08-25", "2017-11-05"), monthly("2008-01-15", "2017-12-15"))))
timings$prtdb_brgdp_per_vintage <- summarise_vintages(BRGDP$base, BRGDP$delay, v_br, "brgdp")

v_ny <- sort(unique(c(daily("2016-11-20", "2017-01-31"), monthly("2005-01-10", "2016-12-10"))))
timings$prtdb_nyfed_per_vintage <- summarise_vintages(NYFED$base, NYFED$legend$delay, v_ny,
                                                      "nyfed")

record_timings(timings)
message("PRTDB fixtures written to ", DIR_R)
