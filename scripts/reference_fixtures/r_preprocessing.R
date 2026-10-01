# Reference outputs of the pre-processing functions of the R package "nowcasting"
# (black box): Bpanel (transformations 0-7, with and without its outlier / missing-value
# correction, Mariano-Murasawa aggregation), month2qtr and qtr2month.
#
# Run from the repository root:  Rscript scripts/reference_fixtures/r_preprocessing.R

source(file.path("scripts", "reference_fixtures", "_common.R"))

data(USGDP)
data(NYFED)
data(BRGDP)
timings <- list()

quiet <- function(expr) suppressWarnings(suppressMessages(expr))

# --- 1. pure transformations (no NA / outlier correction, nothing dropped) ----------
# BRGDP with its own codes (0-5)
t1 <- timed(quiet(Bpanel(BRGDP$base, BRGDP$trans, NA.replace = FALSE, na.prop = 1, h = 0)), 3)
write_ts_csv(t1$value, file.path(DIR_R, "bpanel_transform_brgdp.csv.gz"))
timings$bpanel_transform_brgdp <- t1$seconds

# every code 0-7 on strictly positive BRGDP series (two series per code)
positive <- which(apply(BRGDP$base, 2, function(z) all(z > 0, na.rm = TRUE)))
codes_cols <- positive[seq_len(16)]
codes <- rep(0:7, 2)
codes_in <- BRGDP$base[, codes_cols]
t_codes <- quiet(Bpanel(codes_in, codes, NA.replace = FALSE, na.prop = 1, h = 0))
write_ts_csv(t_codes, file.path(DIR_R, "bpanel_transform_codes.csv.gz"))
write.csv(data.frame(name = colnames(codes_in), transformation = codes),
          file.path(DIR_R, "bpanel_transform_codes_legend.csv"), row.names = FALSE, quote = FALSE)

# NYFED with its codes: monthly 0/1/2 and quarterly 6/7 stored in the 3rd month
t_ny <- quiet(Bpanel(NYFED$base, NYFED$legend$Transformation, NA.replace = FALSE,
                     na.prop = 1, h = 0))
write_ts_csv(t_ny, file.path(DIR_R, "bpanel_transform_nyfed.csv.gz"))

# quarterly series with the monthly codes 1/2 (lag of one MONTH on the monthly grid)
q_in <- NYFED$base[, c("PAYEMS", "UNRATE", "GDPC1", "ULCNFB")]
q12 <- quiet(Bpanel(q_in, c(2, 2, 1, 2), NA.replace = FALSE, na.prop = 1, h = 0))
write_json_fixture(
  list(input = colnames(q_in), codes = c(2, 2, 1, 2), output_columns = colnames(q12),
       n_observed = as.list(setNames(colSums(!is.na(q12)), colnames(q12)))),
  "bpanel_quarterly_monthly_codes.json"
)

# --- 2. Bpanel defaults: outlier + missing-value correction ------------------------
t2 <- timed(quiet(Bpanel(BRGDP$base, BRGDP$trans, h = 0)), 3)
write_ts_csv(column_subset(t2$value, 4), file.path(DIR_R, "bpanel_default_brgdp_subset.csv.gz"))
write_json_fixture(list(kept = colnames(t2$value)), "bpanel_default_brgdp_columns.json")
timings$bpanel_default_brgdp <- t2$seconds

gdp_position <- which(colnames(USGDP$base) == "RGDPGR")
x_us <- USGDP$base[, -gdp_position]
tr_us <- USGDP$legend$Transformation[-gdp_position]
# (the USGDP outputs of Bpanel are written by r_two_step.R, where they are model inputs)
t3 <- timed(quiet(Bpanel(x_us, tr_us, aggregate = FALSE)), 1)
timings$bpanel_default_usgdp <- t3$seconds
t3a <- timed(quiet(Bpanel(x_us, tr_us, aggregate = TRUE)), 1)
timings$bpanel_aggregate_usgdp <- t3a$seconds

# --- 3. Mariano-Murasawa aggregation without corrections ---------------------------
t4 <- quiet(Bpanel(BRGDP$base, BRGDP$trans, NA.replace = FALSE, na.prop = 1, h = 0,
                   aggregate = TRUE))
write_ts_csv(column_subset(t4, 4), file.path(DIR_R, "bpanel_aggregate_brgdp_subset.csv.gz"))

# --- 4. month2qtr / qtr2month -------------------------------------------------------
# month2qtr(reference_month = "mean") only works on a univariate ts in nowcasting 1.1.2,
# so every option is applied series by series. One series with leading missing values
# and one with an incomplete last quarter are included.
na_lead <- which(is.na(BRGDP$base[1, ]) & colSums(!is.na(BRGDP$base)) > 100)[1]
m_names <- unique(c("PMC_VEIC", "PIM_BI", colnames(BRGDP$base)[na_lead]))
for (ref in list(1, 2, 3, "mean")) {
  cols <- lapply(m_names, function(nm) month2qtr(BRGDP$base[, nm], reference_month = ref))
  out <- do.call(cbind, cols)
  colnames(out) <- m_names
  write_ts_csv(out, file.path(DIR_R, sprintf("month2qtr_%s.csv", ref)))
}
for (ref in 1:3) {
  for (interp in c(FALSE, TRUE)) {
    out <- qtr2month(BRGDP$GDP, reference_month = ref, interpolation = interp)
    write_ts_csv(out, file.path(DIR_R, sprintf("qtr2month_%d_%s.csv", ref, tolower(interp))))
  }
}

record_timings(timings)
message("preprocessing fixtures written to ", DIR_R)
