# Shared helpers of the reference-fixture scripts (R side).
#
# Clean-room rule (plan section 11): the GPL-3 package "nowcasting" is used ONLY as a
# black box. These scripts call its exported functions (Bpanel, ICfactors, ICshocks,
# nowcast, PRTDB, month2qtr, qtr2month) and its bundled datasets, and write their
# inputs and outputs to tests/reference_validation/fixtures/. Nothing here reads,
# copies or translates the package source code.
#
# Run from the repository root:  Rscript scripts/reference_fixtures/<script>.R

suppressPackageStartupMessages({
  library(nowcasting)
  library(jsonlite)
})

FIXTURES <- file.path("tests", "reference_validation", "fixtures")
DIR_INPUTS <- file.path(FIXTURES, "inputs")
DIR_R <- file.path(FIXTURES, "r")
dir.create(DIR_INPUTS, recursive = TRUE, showWarnings = FALSE)
dir.create(DIR_R, recursive = TRUE, showWarnings = FALSE)

# Monthly "YYYY-MM" labels of a monthly ts / mts.
month_labels <- function(x) {
  tt <- time(x)
  year <- floor(tt + 1e-8)
  month <- round((tt - year) * 12) + 1
  sprintf("%04d-%02d", as.integer(year), as.integer(month))
}

# Quarterly "YYYYQq" labels of a quarterly ts / mts.
quarter_labels <- function(x) {
  tt <- time(x)
  year <- floor(tt + 1e-8)
  q <- round((tt - year) * 4) + 1
  sprintf("%04dQ%d", as.integer(year), as.integer(q))
}

period_labels <- function(x) {
  if (frequency(x) == 12) month_labels(x) else quarter_labels(x)
}

# Format numbers losslessly (17 significant digits; NA -> empty field).
fmt_num <- function(v, digits = 17) {
  out <- ifelse(is.na(v), "", formatC(v, digits = digits, format = "g"))
  trimws(out)
}

# Write a ts / mts (or matrix with explicit labels) as a gzip CSV with a "period" column.
write_ts_csv <- function(x, path, labels = NULL, digits = 17) {
  m <- as.matrix(x)
  if (is.null(colnames(m))) colnames(m) <- paste0("V", seq_len(ncol(m)))
  if (is.null(labels)) labels <- period_labels(x)
  df <- data.frame(period = labels, stringsAsFactors = FALSE)
  for (j in seq_len(ncol(m))) df[[colnames(m)[j]]] <- fmt_num(m[, j], digits)
  con <- if (grepl("\\.gz$", path)) gzfile(path, "w") else file(path, "w")
  on.exit(close(con))
  write.csv(df, con, row.names = FALSE, quote = FALSE, na = "")
  invisible(path)
}

# Read a "period"-indexed monthly CSV written by the Python side as an mts.
read_monthly_csv <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  first <- df$period[1]
  start <- c(as.integer(substr(first, 1, 4)), as.integer(substr(first, 6, 7)))
  ts(as.matrix(df[, -1, drop = FALSE]), start = start, frequency = 12)
}

# Every `by`-th column (keeps the fixtures small; the tests rebuild full panels with an
# emulation of Bpanel validated on these subsets).
column_subset <- function(x, by = 10) x[, seq(1, ncol(x), by = by), drop = FALSE]

# Append `n` empty (NA) months to an mts (forecast horizon).
pad_rows <- function(x, n) {
  out <- ts(rbind(as.matrix(x), matrix(NA_real_, n, ncol(x))), start = start(x),
            frequency = frequency(x))
  colnames(out) <- colnames(x)
  out
}

write_json_fixture <- function(obj, name) {
  path <- file.path(DIR_R, name)
  writeLines(toJSON(obj, auto_unbox = TRUE, digits = NA, null = "null", na = "null",
                    pretty = TRUE), path)
  invisible(path)
}

# Elapsed wall-clock seconds of an expression (median of `reps` runs) and its value.
timed <- function(expr, reps = 1) {
  f <- substitute(expr)
  env <- parent.frame()
  times <- numeric(reps)
  value <- NULL
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    value <- eval(f, env)
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  list(value = value, seconds = stats::median(times))
}

r_session_info <- function() {
  list(
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    nowcasting_version = as.character(utils::packageVersion("nowcasting")),
    generated = format(Sys.time(), "%Y-%m-%d"),
    platform = R.version$platform
  )
}

# Append / replace timing entries in fixtures/r/timings.json.
record_timings <- function(entries) {
  path <- file.path(DIR_R, "timings.json")
  current <- if (file.exists(path)) fromJSON(path, simplifyVector = FALSE) else list()
  for (k in names(entries)) current[[k]] <- entries[[k]]
  writeLines(toJSON(current, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
  invisible(path)
}
